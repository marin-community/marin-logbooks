# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ensure the launcher's snapshots and check modified-input sandboxes before serving."""

import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from rigging.secrets import require_secret
from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL, DaytonaConnection
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.daytona.snapshot import SnapshotRequest, ensure_snapshot
from shellbox.machine import Command, MachineSpec, NetworkPolicy

ROOT = Path(__file__).resolve().parent


async def main():
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    # Resolve actual Harbor resource enforcement instead of guessing snapshot identities.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT.parents[2] / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"),
            "snapshot-specs",
            str(ROOT / "full.yaml"),
        ],
        check=True,
        stdout=subprocess.PIPE,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT.parents[2] / "lib/marin/src")},
        text=True,
    )
    (output / "snapshot-specs.json").write_text(result.stdout)
    requests = [SnapshotRequest.from_dict(x) for x in json.loads(result.stdout)]
    assert len(requests) == 45
    connection = DaytonaConnection(require_secret("DAYTONA_EVAL_API_KEY"), DEFAULT_DAYTONA_API_URL, "us")
    semaphore = asyncio.Semaphore(4)
    async with connection.client_factory()() as client:

        async def ensure(request):
            async with semaphore:
                name = await ensure_snapshot(client, request)
                print(json.dumps({"snapshot": name, "state": "active"}), flush=True)

        await asyncio.gather(*(ensure(request) for request in requests))
    manifest = json.loads((ROOT / "task-manifest.json").read_text())
    checks = [x for x in manifest if x["input_variant"] or x["task"] in {"da-19-4", "da-20-4"}]
    factory = DaytonaMachineFactory(connection.client_factory(), ttl_minutes=20, create_timeout=300)
    reports = []
    for task in checks:
        machine = None
        task_id = task["task"]
        config = json.loads((ROOT / "runtime-tasks" / task_id / "environment/shellbox-images.json").read_text())
        image = config["main"]["image"]
        request = next(
            x
            for x in requests
            if x.source.reference == image and x.resources.memory * 1024 == task["resources"]["memory_mb"]
        )
        report = {"task": task_id, "snapshot": request.name, "image_digest": task["image_digest"]}
        try:
            machine = await factory.create(
                MachineSpec(
                    source=request.source,
                    workdir="/app",
                    cpus=request.resources.cpu,
                    memory_mb=request.resources.memory * 1024,
                    storage_mb=request.resources.disk * 1024,
                    network=NetworkPolicy.ALLOW,
                    startup_timeout=300,
                )
            )
            hashes = "\n".join(x["sha256"] + "  /app/data/" + x["path"] for x in task["input_files"]) + "\n"
            command = "set -e; pi --version; printf %s " + shlex.quote(hashes) + " | sha256sum -c -; df -h /app"
            if task_id.startswith("da-1-"):
                code = (
                    "import shlex; f=open('/app/data/GSE236581_CRC-ICB_metadata.txt'); fields=shlex.s"
                    "plit(f.readline()); assert {'Patient','SubCellType','Treatment','Tissue'}.issubs"
                    "et(fields); print(fields)"
                )
                command += "; test ! -e /app/data/GSE236581_counts.mtx; python3 -c " + shlex.quote(code)
            if task_id.startswith("da-17-"):
                code = (
                    "import h5py; f=h5py.File('/app/data/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad');"
                    " assert f['X'].id==f['raw/X'].id; assert {'author_cell_type','donor_id','disease"
                    "','self_reported_ethnicity'}.issubset(f['obs']); assert tuple(f['raw/X'].attrs['"
                    "shape'])==(1263676,30172); print(list(f['obs']))"
                )
                command += (
                    "; /opt/biomnibench-judge/bin/pip install --quiet h5py==3.15.1; /opt/biomnibench-"
                    "judge/bin/python -c " + shlex.quote(code)
                )
            result = await machine.run(Command(argv=("bash", "-lc", command), timeout=300))
            report.update(
                exit_code=result.exit_code,
                stdout=result.stdout.decode(),
                stderr=result.stderr.decode(),
                status="passed" if result.exit_code == 0 else "failed",
            )
        finally:
            if machine is not None:
                await machine.close()
                report["cleanup"] = "closed"
            reports.append(report)
            (output / "runtime-report.json").write_text(json.dumps(reports, indent=2) + "\n")
        if report.get("status") != "passed":
            raise RuntimeError("Runtime check failed for " + task_id)
    (output / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "snapshots": len(requests),
                "runtime_checks": len(reports),
                "all_closed": all(x.get("cleanup") == "closed" for x in reports),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
