# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the released router's optional prefix binding against mock DP ranks."""

import argparse
import copy
import hashlib
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from router_process import upstream_router

MODEL = "hero-router-cpu-fixture"
WHEEL_URL = "https://files.pythonhosted.org/packages/54/c0/0bb281992cdc35b5944929794b470d5e3af37fd0ff80ce6d6da2bbcaf6e8/vllm_router-0.1.16-cp38-abi3-manylinux_2_28_x86_64.whl"
WHEEL_SHA256 = "f400a6d7ba80ab613f28b915a7b81ba8cddcb85732470e3c9737d3430ed6d251"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def request(url: str, body: dict | None = None, session: str = "") -> bytes:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "X-Session-ID": session, "X-Request-ID": session}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=15) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    observed = []
    pressure = [0.2] * 4

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def send(self, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/metrics":
                body = "\n".join(
                    f'vllm:{name}{{engine="{rank}",model_name="{MODEL}"}} {value}\n'
                    for rank in range(4)
                    for name, value in (
                        ("kv_cache_usage_perc", pressure[rank]),
                        ("num_requests_running", 0),
                        ("num_requests_waiting", 0),
                        ("prefix_cache_queries_total", 0),
                        ("prefix_cache_hits_total", 0),
                    )
                )
                self.send(body.encode(), "text/plain")
            elif self.path == "/v1/models":
                self.send(json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model"}]}).encode())
            else:
                self.send(b"{}")

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            rank = int(self.headers["X-data-parallel-rank"])
            observed.append(
                {"session": self.headers["X-Session-ID"], "rank": rank, "body": body, "body_sha256": digest(body)}
            )
            response = {
                "id": "cpu-fixture",
                "object": "chat.completion",
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "fixture"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 1, "total_tokens": 2001},
            }
            self.send(json.dumps(response).encode())

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    base = json.loads(args.manifest.read_text())["router"]
    base |= {
        "wheel": {"url": WHEEL_URL, "sha256": WHEEL_SHA256},
        "backend_url": f"http://127.0.0.1:{backend.server_port}",
        "base_url": "http://127.0.0.1:39940",
        "data_parallel_size": 4,
    }
    cases = [(f"prefix-peer-{i}", "shared context " * 2048 + f"branch {i}") for i in range(16)]
    cases.extend(
        [("prefix-peer-0", "shared context " * 2048 + "continued history"), ("isolated", "unrelated text " * 2048)]
    )
    results = {}
    try:
        for mode in ("native_hash", "binding_hash", "binding_cache"):
            config = copy.deepcopy(base)
            if mode != "native_hash":
                config["program_binding"] = {
                    "binding_only": True,
                    "program_scheduling_enable_key": "auto",
                    "binding_strategy": "cache_aware" if mode == "binding_cache" else "consistent_hash",
                }
            output = args.output / mode
            output.mkdir()
            start = len(observed)
            with upstream_router(config, output) as url:
                workers = json.loads(request(url + "/workers"))
                for session, content in cases:
                    body = {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": content}],
                        "stream": False,
                        "max_tokens": 1,
                    }
                    response = json.loads(request(url + "/v1/chat/completions", body, session))
                    assert response["choices"][0]["message"]["content"] == "fixture"
                    assert observed[-1]["body"]["messages"] == body["messages"]
                diagnostics = None
                if mode != "native_hash":
                    diagnostics = json.loads(request(url + "/scheduling/diagnostics"))
                    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
            rows = observed[start:]
            assert len(rows) == len(cases)
            assert all(0 <= row["rank"] < 4 for row in rows)
            results[mode] = {"config": config, "workers": workers, "requests": rows, "diagnostics": diagnostics}
        native = results["native_hash"]["requests"]
        hashed = results["binding_hash"]["requests"]
        cached = results["binding_cache"]["requests"]
        assert len({row["rank"] for row in native[:16]}) > 1
        assert [row["rank"] for row in native] == [row["rank"] for row in hashed]
        assert len({row["rank"] for row in cached[:17]}) == 1
        assert [row["body_sha256"] for row in native] == [row["body_sha256"] for row in hashed]
        assert [row["body_sha256"] for row in native] == [row["body_sha256"] for row in cached]
        result = {
            "scope": (
                "CPU-only real released x86 Router0.1.16 CLI with four mocked concrete DP ranks. No model, GPU, "
                "numerical correctness or throughput claim. Mock KV pressure is equal and fixed at0.2. Tests prefix "
                "co-location across different conversation IDs, same-ID continuation, preserved hash control and "
                "identical forwarded JSON bodies."
            ),
            "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "results": results,
            "status": "passed",
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "requests_per_mode": len(cases),
                    "native_ranks": [row["rank"] for row in native],
                    "cache_ranks": [row["rank"] for row in cached],
                }
            )
        )
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
