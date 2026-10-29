# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned upstream HTTP router beside an already ready native server."""

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def upstream_router(config: dict, output: Path):
    requirements = Path(__file__).with_name(config["requirements_file"])
    assert hashlib.sha256(requirements.read_bytes()).hexdigest() == config["requirements_sha256"]
    wheel = config["wheel"]
    port = config["base_url"].rsplit(":", 1)[1]
    command = [
        "uvx",
        "--python",
        "3.12",
        "--from",
        f"vllm-router @ {wheel['url']}#sha256={wheel['sha256']}",
        "--with-requirements",
        str(requirements),
        "vllm-router",
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--worker-urls",
        config["backend_url"],
        "--policy",
        config["policy"],
        "--intra-node-data-parallel-size",
        str(config["data_parallel_size"]),
        "--request-timeout-secs",
        str(config["request_timeout_secs"]),
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(int(port) + 1),
    ]
    assert config["disable_retries"] is True
    command.append("--disable-retries")
    (output / "router-command.json").write_text(json.dumps(command, indent=2) + "\n")
    environment = {key: value for key, value in os.environ.items() if key != "UV_CONSTRAINT"}
    environment["UV_NO_CONFIG"] = "1"
    with (output / "router.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while True:
                assert process.poll() is None, f"Router exited: {process.returncode}; see {output / 'router.log'}"
                try:
                    with urllib.request.urlopen(config["base_url"] + "/health", timeout=3) as response:
                        assert response.status == 200
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Upstream router did not become healthy")
                    time.sleep(1)
            with urllib.request.urlopen(config["base_url"] + "/workers", timeout=10) as response:
                (output / "router-workers.json").write_bytes(response.read())
            yield config["base_url"]
            assert process.poll() is None, f"Router exited during requests: {process.returncode}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
