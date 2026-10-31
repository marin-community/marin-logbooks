# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the installed native SM100 CUTLASS cache on one batch GB200 node.

The allocation is four GPUs and must be counted in full, although compilation
hides CUDA devices and starts no model. A controller-side placement check must
admit the actual task attempt before the bounded native build starts.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.agentx.artifacts import retain_artifacts
from experiments.evaluation.agentx.processes import run_logged
from experiments.evaluation.agentx.run import interrupt


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(manifest: dict) -> None:

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    info = get_job_info()
    assert info is not None and info.num_tasks == 1 and info.task_index == 0
    node = os.environ["IRIS_NODE_NAME"]
    assert node in manifest["placement"]["nodes"]
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).split()
    assert len(uuids) == 4 and set(uuids).isdisjoint(manifest["excluded_gpu_uuids"])
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    record = {
        "status": "awaiting_admission",
        "started_unix": time.time(),
        "job_id": str(info.job_id),
        "task_id": os.environ["IRIS_TASK_ID"],
        "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "node": node,
        "gpu_uuids": uuids,
        "allocated_gpus": 4,
        "cuda_visible_devices": "",
        "manifest": manifest,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = output / "preparation.json"
    write(path, record)
    try:
        deadline = time.monotonic() + 600
        gate = output / "admission.json"
        while not gate.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Fresh placement/physical-attempt admission did not arrive")
            time.sleep(1)
        admission = json.loads(gate.read_text())
        assert admission["task_id"] == record["task_id"] and admission["attempt_uid"] == record["attempt_uid"]
        assert admission["node"] == node and set(admission["gpu_uuids"]) == set(uuids)
        assert admission["hero_training_gpus"] == 0 and admission["physical_groups_disjoint"]
        assert time.time() - admission["checked_unix"] < 120
        cpus = sorted(os.sched_getaffinity(0))[-8:]
        assert len(cpus) == 8
        os.sched_setaffinity(0, cpus)
        os.nice(10)
        environment = dict(os.environ) | {
            "CUDA_VISIBLE_DEVICES": "",
            "FLASHINFER_CUDA_ARCH_LIST": "10.0a",
            "FLASHINFER_WORKSPACE_BASE": manifest["flashinfer_workspace_base"],
            "MAX_JOBS": "8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "UV_NO_CONFIG": "1",
            "UV_CONSTRAINT": "/app/experiments/evaluation/agentx/requirements-vllm-hero.txt",
        }
        record.update(status="preflighting", cpu_affinity=cpus, admitted=admission)
        write(path, record)
        for phase, limit in [("preflight", 180), ("build", 7200)]:
            command = [
                *manifest["serving_prefix"],
                *manifest["cuda_bootstrap"],
                "python",
                str(Path(__file__).with_name("cutlass_prepare_worker.py").resolve()),
                "--phase",
                phase,
                "--output",
                str(output),
            ]
            write(output / (phase + "-command.json"), {"argv": command})
            run_logged(command, environment, output / (phase + ".log"), timeout=limit)
            record[phase] = json.loads((output / (phase + "-worker.json")).read_text())
            record["status"] = "building" if phase == "preflight" else "retaining_cache"
            write(path, record)
        assert record["preflight"]["ninja_sha256"] == record["build"]["ninja_sha256"]
        workspace = Path(manifest["flashinfer_workspace_base"])
        bundle = output.parent / ("cutlass-cache-" + record["attempt_uid"] + ".tar.zst")
        prefix = manifest["retention_prefix"] + "/" + record["attempt_uid"]
        uri, digest = retain_artifacts(workspace, bundle, prefix, name="flashinfer-workspace")
        filesystem, key = filesystem_for(uri)
        with filesystem.open(key, "rb") as stream:
            readback = hashlib.file_digest(stream, "sha256").hexdigest()
        assert readback == digest
        record.update(
            status="native_cpu_build_cache_retained",
            archive_uri=uri,
            archive_sha256=digest,
            readback_sha256=readback,
            archive_bytes=bundle.stat().st_size,
        )
        write(path, record)
        if component := manifest.get("component"):
            command = [
                *manifest["serving_prefix"],
                *manifest["cuda_bootstrap"],
                "python",
                str(Path(__file__).with_name("moe_output_probe.py").resolve()),
                *component,
                "--output",
                str(output / "component/result.json"),
            ]
            write(output / "component-command.json", {"argv": command})
            record["status"] = "checking_one_gpu_component"
            write(path, record)
            run_logged(command, environment | {"CUDA_VISIBLE_DEVICES": "0"}, output / "component.log", timeout=900)
            record["component"] = json.loads((output / "component/result.json").read_text())
            record["status"] = "native_cache_retained_and_component_passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        workspace = Path(manifest["flashinfer_workspace_base"])
        if "archive_uri" not in record and workspace.exists():
            try:
                bundle = output.parent / ("cutlass-partial-" + record["attempt_uid"] + ".tar.zst")
                prefix = manifest["retention_prefix"] + "/" + record["attempt_uid"]
                uri, digest = retain_artifacts(workspace, bundle, prefix, name="partial-workspace")
                filesystem, key = filesystem_for(uri)
                with filesystem.open(key, "rb") as stream:
                    readback = hashlib.file_digest(stream, "sha256").hexdigest()
                assert readback == digest
                record["partial_retention"] = {
                    "archive_uri": uri,
                    "archive_sha256": digest,
                    "readback_sha256": readback,
                    "archive_bytes": bundle.stat().st_size,
                }
            except Exception as retention_error:
                record["partial_retention_error"] = type(retention_error).__name__
        raise
    finally:
        record["finished_unix"] = time.time()
        write(path, record)
    print(json.dumps({key: value for key, value in record.items() if key != "manifest"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.manifest.read_text()))


if __name__ == "__main__":
    main()
