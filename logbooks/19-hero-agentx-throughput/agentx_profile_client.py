# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture five native profiler steps during a separate, short AgentX replay.

Keep the native workload and count-driven warmup. The 900-second scoring
window is an intrusive diagnostic and cannot supply a full-hour result.
"""

import argparse
import hashlib
import json
import tempfile
import time
import urllib.request
from pathlib import Path

from agentx_candidate import SESSION_HEADER_ENV, pack_metrics_export
from cache_probe import batch_trace
from router_process import upstream_router

from experiments.evaluation.agentx.processes import logged_process, stop_group
from experiments.evaluation.agentx.recipe import benchmark_command, load_series, workload_environment
from experiments.evaluation.agentx.workload import check_native_cache, restore_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    series = load_series(args.series)
    assert hashlib.sha256(args.series.read_bytes()).hexdigest() == manifest["series_sha256"]
    diagnostic = manifest["diagnostic_request"]["agentx_profile"]
    assert diagnostic == {"score_seconds": 900, "trigger_after_seconds": 120}
    assert manifest["diagnostic_request"]["trace_batches"] is True
    root = args.output
    record = {
        "scope": "Intrusive 900-second AgentX diagnostic; no full-hour throughput claim",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "source": manifest["source"],
        "status": "starting",
        "events": [],
    }
    process = None
    try:
        deadline = time.monotonic() + series["ready_timeout"]
        while True:
            path = root / "diagnostic.json"
            server = json.loads(path.read_text()) if path.exists() else {"status": "starting"}
            if server["status"] == "ready":
                break
            assert server["status"] == "starting", server["status"]
            assert time.monotonic() < deadline, "Diagnostic server readiness deadline exceeded"
            time.sleep(5)
        config = server["profiler_config"]
        assert server["profiler_configured"] and config["max_iterations"] == 5
        assert config["active_iterations"] == 5
        assert not config["torch_profiler_with_stack"] and not config["torch_profiler_record_shapes"]
        point = root / "point"
        point.mkdir(exist_ok=False)
        short_series = series | {"duration": diagnostic["score_seconds"]}
        router = manifest["router"]
        storage = batch_trace(router["backend_url"], "hero_cache_storage")
        (point / "kv-storage-first-engine.json").write_text(json.dumps(storage, indent=2) + "\n")
        command = benchmark_command(
            short_series, manifest["concurrency"], router["base_url"], point / "aiperf_artifacts"
        )
        record.update(command=command, profiler_config=config, started_unix=time.time())
        with tempfile.TemporaryDirectory(prefix="agentx-profile-workload-", dir="/tmp") as temporary:
            bundle = Path(temporary) / "bundle"
            restore_workload(manifest["workload"], bundle)
            check_native_cache(bundle, args.series, point / "workload-check.log")
            environment = workload_environment(bundle, short_series) | {SESSION_HEADER_ENV: "1"}
            log = point / "agentx.log"
            with upstream_router(router, point):
                process = logged_process(command, environment, log)
                # The pinned warmup is count driven. Its grace setting does
                # not cap the time needed to warm all session-tree lanes.
                deadline = time.monotonic() + series["task_timeout"]
                while not log.exists() or "Credit phase start: profiling" not in log.read_text(errors="replace"):
                    assert process.poll() is None, "AgentX ended before the scoring phase"
                    assert time.monotonic() < deadline, "Diagnostic warmup deadline exceeded"
                    time.sleep(5)
                record["events"].append({"event": "scoring_observed", "unix": time.time()})
                batch_trace(router["backend_url"], "install_hero_batch_trace")
                record["events"].append({"event": "dispatch_trace_installed", "unix": time.time()})
                trigger = time.monotonic() + diagnostic["trigger_after_seconds"]
                while time.monotonic() < trigger:
                    assert process.poll() is None, "AgentX ended before the profile trigger"
                    time.sleep(1)
                record["events"].append({"event": "start_profile_requested", "unix": time.time()})
                request = urllib.request.Request(router["backend_url"] + "/start_profile", data=b"", method="POST")
                with urllib.request.urlopen(request, timeout=120) as response:
                    assert response.status == 200
                    record["events"].append({"event": "start_profile_returned", "unix": time.time()})
                assert process.wait(timeout=max(1, deadline - time.monotonic())) == 0
                traces = batch_trace(router["backend_url"], "take_hero_batch_trace")
                (point / "dispatch-first-engine.json").write_text(json.dumps(traces, indent=2) + "\n")
                batch_trace(router["backend_url"], "stop_hero_batch_trace")
                record["dispatch_trace_scope"] = (
                    "CPU dispatch metadata during this short diagnostic. The RPC returns only first-engine results; "
                    "each GPU worker independently saves its own trace and KV storage record under IRIS_OUTPUT_DIR. "
                    "Dispatch counts are not time shares or FLOPs."
                )
            export = json.loads((point / "aiperf_artifacts/profile_export_aiperf.json").read_text())
            assert not export["was_cancelled"]
            record.update(status="completed", native_metadata=export["metadata"])
        pack_metrics_export(root)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if process is not None:
            stop_group(process)
        record["finished_unix"] = time.time()
        root.mkdir(parents=True, exist_ok=True)
        (root / "agentx-profile-client.json").write_text(json.dumps(record, indent=2) + "\n")
        (root / "stop").touch()


if __name__ == "__main__":
    main()
