# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise both real profile inspectors against a local metadata RPC boundary."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

DIRECTORY = Path(__file__).parent


def main():
    state = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["request"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"results": [state["response"]]}).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    record = {
        "scope": (
            "CPU protocol preflight of the actual inspector commands and profiles using a local fake RPC producer. "
            "No loaded-weight, dtype, KV or GPU correctness claim. Actual native storage gates remain required."
        ),
        "status": "running",
        "cases": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="hero-triton-inspector-") as temporary:
            root = Path(temporary)
            output = root / "correctness" / "inspection.json"
            output.parent.mkdir()
            for name, ignored in (
                ("hybrid-fp8-24of48-8gpu.yaml", [*range(12), *range(36, 48)]),
                ("hybrid-fp8-16of48-8gpu.yaml", [*range(16), *range(32, 48)]),
            ):
                profile_path = DIRECTORY / name
                profile = yaml.safe_load(profile_path.read_text())
                config = root / "profile.json"
                config.write_text(json.dumps(profile))
                command = next(check["command"] for check in profile["checks"] if check["kind"] == "serving")
                assert command[0] == "python"
                hashes = {
                    "extension_sha256": hashlib.sha256((DIRECTORY / "hero_hybrid_metadata.py").read_bytes()).hexdigest(),
                    "helper_sha256": hashlib.sha256((DIRECTORY / "hero_weight_metadata.py").read_bytes()).hexdigest(),
                }
                for perturbation in ("matching", "wrong_layer_count", "wrong_producer_hash"):
                    response = {
                        "status": "passed",
                        **hashes,
                        "dp_size": profile["gpus"]["count"],
                        "tp_size": 1,
                        "expected_bf16_layers": ignored,
                        "bf16_expert_layers": len(ignored),
                        "fp8_expert_layers": 48 - len(ignored),
                    }
                    if perturbation == "wrong_layer_count":
                        response["fp8_expert_layers"] += 1
                    elif perturbation == "wrong_producer_hash":
                        response["extension_sha256"] = "0" * 64
                    state["response"] = response
                    result = subprocess.run(
                        [sys.executable, *command[1:]],
                        env=os.environ
                        | {
                            "KNOTMETER_PROFILE": str(config),
                            "KNOTMETER_SERVER_URL": f"http://127.0.0.1:{server.server_port}/v1",
                            "KNOTMETER_RUNTIME_SOURCE": profile["vllm"]["revision"],
                            "KNOTMETER_CHECK_OUTPUT": str(output),
                        },
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    report = json.loads(output.read_text())
                    assert state["request"]["method"] == "inspect_hero_hybrid_storage"
                    assert state["request"]["args"] == [ignored, "correctness"]
                    assert report["expected_bf16_layers"] == ignored
                    assert report["expected_fp8_layers"] == [i for i in range(48) if i not in ignored]
                    assert (result.returncode == 0) == (perturbation == "matching")
                    assert report["status"] == ("passed" if perturbation == "matching" else "failed")
                    record["cases"].append(
                        {
                            "profile": name,
                            "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                            "perturbation": perturbation,
                            "requested_bf16_layers": ignored,
                            "expected_fp8_layers": 48 - len(ignored),
                            "exit_code": result.returncode,
                            "status": report["status"],
                            "request": state["request"],
                        }
                    )
        record["status"] = "passed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        (DIRECTORY / "hybrid-inspector-cpu-preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "cases": len(record["cases"])}))


if __name__ == "__main__":
    main()
