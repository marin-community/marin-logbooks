# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start the native profiler during a long-context diagnostic's decoding phase."""

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    record = {"scope": "Intrusive native diagnostic; not AgentX or throughput confirmation", "events": []}
    process = None
    try:
        deadline = time.monotonic() + 900
        while True:
            server = json.loads((root / "diagnostic.json").read_text())
            if server["status"] == "ready":
                break
            assert server["status"] == "starting", server["status"]
            assert time.monotonic() < deadline, "Diagnostic server readiness deadline exceeded"
            time.sleep(5)
        assert server["profiler_configured"] and server["profiler_config"]["max_iterations"] == 5
        command = server["server_command"]
        prefix = command[: command.index("python")]
        cases = json.loads(Path(__file__).with_name("experiments").joinpath("native-diagnostics.json").read_text())
        argv = next(case["argv"] for case in cases["cases"] if case["name"] == "long64k-c16").copy()
        for flag, value in (("--random-output-len", 1024), ("--max-concurrency", 64), ("--num-prompts", 64)):
            argv[argv.index(flag) + 1] = str(value)
        client = root / "steady-client"
        client.mkdir(exist_ok=False)
        # The server's native readiness check has passed. Avoid profiling a
        # separate initial 1,024-token smoke request before the concurrent bank.
        argv += ["--ready-check-timeout-sec", "0", "--result-dir", str(client), "--result-filename", "result.json"]
        environment = os.environ | {
            "UV_NO_CONFIG": "1",
            "UV_CONSTRAINT": "/app/experiments/evaluation/agentx/requirements-vllm-hero.txt",
        }
        record.update(command=prefix + argv, server_source=server["source"], started_unix=time.time())
        with (client / "client.log").open("w") as log:
            process = subprocess.Popen(record["command"], env=environment, stdout=log, stderr=subprocess.STDOUT)
            time.sleep(100)
            assert process.poll() is None, "Diagnostic bank ended before the profiling trigger"
            record["events"].append({"event": "start_profile_requested", "unix": time.time()})
            request = urllib.request.Request("http://127.0.0.1:38197/start_profile", data=b"", method="POST")
            with urllib.request.urlopen(request, timeout=120) as response:
                record["events"].append(
                    {"event": "start_profile_returned", "unix": time.time(), "status": response.status}
                )
            assert process.wait(timeout=900) == 0
        result = json.loads((client / "result.json").read_text())
        assert result["completed"] == 64 and result["failed"] == 0
        record.update(status="completed", finished_unix=time.time(), result=result)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        (root / "steady-profile-client.json").write_text(json.dumps(record, indent=2) + "\n")
        (root / "stop").touch()


if __name__ == "__main__":
    main()
