# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the released router's optional prefix binding against mock DP ranks."""

import argparse
import copy
import hashlib
import json
import platform
import threading
import time
import urllib.request
import uuid
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
    headers = {"Content-Type": "application/json", "X-Session-ID": session, "X-Request-ID": str(uuid.uuid4())}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=15) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-affinity-retention", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    observed = []
    pressure = [0.2] * 4
    http_counts = {}

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
            key = "GET " + self.path
            http_counts[key] = http_counts.get(key, 0) + 1
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
            key = "POST " + self.path
            http_counts[key] = http_counts.get(key, 0) + 1
            assert self.path == "/v1/chat/completions"
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
            if body["stream"]:
                chunks = [
                    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "fixture"}}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": response["usage"]},
                ]
                data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                self.send(data.encode(), "text/event-stream")
            else:
                self.send(json.dumps(response).encode())

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    base = json.loads(args.manifest.read_text())["router"]
    architecture = platform.machine()
    if architecture == "x86_64":
        base["wheel"] = {"url": WHEEL_URL, "sha256": WHEEL_SHA256}
    else:
        assert architecture == "aarch64"
        assert base["wheel"]["sha256"] == "be6137553ef807f0b3303194da9eb1dfdf65913ac1b8a84c4ea0a7a62683cfec"
    base |= {
        "backend_url": f"http://127.0.0.1:{backend.server_port}",
        "base_url": "http://127.0.0.1:39940",
        "data_parallel_size": 4,
    }
    shared = "shared context " * 2048
    cases = [(f"prefix-peer-{i}", shared + f"branch {i}", False, None) for i in range(16)]
    cases.extend(
        [
            ("prefix-peer-0", shared + "continued history", False, None),
            ("isolated", "unrelated text " * 2048, False, None),
            ("moderate-pressure", shared + "moderate", False, [0.35, 0.2, 0.2, 0.2]),
            ("high-pressure", shared + "high", False, [0.95, 0.2, 0.2, 0.2]),
            ("prefix-peer-0", shared + "bound continuation", False, None),
            ("stream-peer", shared + "streaming", True, [0.2] * 4),
        ]
    )

    def wait_observed(url: str) -> dict:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            diagnostics = json.loads(request(url + "/scheduling/diagnostics"))
            ranks = diagnostics["ranks"]
            if len(ranks) == 4 and all(
                rank["observation_fresh"]
                and rank["vllm_kv_cache_usage"] is not None
                and abs(rank["vllm_kv_cache_usage"] - pressure[rank["dp_rank"]]) < 1e-8
                for rank in ranks
            ):
                return diagnostics
            time.sleep(0.05)
        raise AssertionError("Router did not observe all four mock KV pressures within five seconds")

    results = {}
    try:
        for mode in ("native_hash", "binding_hash", "binding_cache"):
            config = copy.deepcopy(base)
            pressure[:] = [0.2] * 4
            if mode != "native_hash":
                config["program_binding"] = {
                    "binding_only": True,
                    "program_scheduling_enable_key": "auto",
                    "binding_strategy": "cache_aware" if mode == "binding_cache" else "consistent_hash",
                    "token_capacity_per_dp_rank": 1_000_000,
                }
            output = args.output / mode
            output.mkdir()
            start = len(observed)
            retention = None
            with upstream_router(config, output) as url:
                workers = json.loads(request(url + "/workers"))
                pressure_observations = []
                for case_index, (session, content, stream, next_pressure) in enumerate(cases):
                    if next_pressure is not None:
                        pressure[:] = next_pressure
                        if mode != "native_hash":
                            pressure_observations.append(wait_observed(url))
                    body = {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": content}],
                        "stream": stream,
                        "max_tokens": 1,
                    }
                    response = request(url + "/v1/chat/completions", body, session)
                    if stream:
                        assert b"fixture" in response and response.endswith(b"data: [DONE]\n\n")
                    else:
                        assert json.loads(response)["choices"][0]["message"]["content"] == "fixture"
                    assert observed[-1]["body"]["messages"] == body["messages"]
                    if case_index == 0 and mode != "native_hash":
                        pressure_observations.append(wait_observed(url))
                diagnostics = None
                if mode != "native_hash":
                    diagnostics = wait_observed(url)
                    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
            rows = observed[start : start + len(cases)]
            assert len(rows) == len(cases)
            assert all(0 <= row["rank"] < 4 for row in rows)
            results[mode] = {
                "config": config,
                "workers": workers,
                "requests": rows,
                "diagnostics": diagnostics,
                "pressure_observations": pressure_observations,
                "affinity_retention": retention,
            }
        native = results["native_hash"]["requests"]
        hashed = results["binding_hash"]["requests"]
        cached = results["binding_cache"]["requests"]
        assert len({row["rank"] for row in native[:16]}) > 1
        assert [row["rank"] for row in native] == [row["rank"] for row in hashed]
        assert len({row["rank"] for row in cached[:17]}) == 1
        assert cached[-4]["rank"] == cached[0]["rank"]
        assert cached[-3]["rank"] != cached[0]["rank"]
        assert cached[-2]["rank"] == cached[0]["rank"]
        assert [row["body_sha256"] for row in native] == [row["body_sha256"] for row in hashed]
        assert [row["body_sha256"] for row in native] == [row["body_sha256"] for row in cached]
        retention = None
        if args.test_affinity_retention:
            pressure[:] = [0.2] * 4
            retention_output = args.output / "affinity_retention"
            retention_output.mkdir()
            config = copy.deepcopy(results["binding_cache"]["config"])
            extra_start = len(observed)
            with upstream_router(config, retention_output) as url:
                wait_observed(url)

                def send_retention_case(session: str, content: str) -> int:
                    body = {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": content}],
                        "stream": False,
                        "max_tokens": 1,
                    }
                    request(url + "/v1/chat/completions", body, session)
                    rows = observed[extra_start:]
                    (retention_output / "requests.json").write_text(json.dumps(rows, indent=2) + "\n")
                    expected = results["native_hash"]["requests"][0]["body"] | {"messages": body["messages"]}
                    assert observed[-1]["body_sha256"] == digest(expected)
                    return observed[-1]["rank"]

                long_prefix = "independent delayed long context " * 1000
                short_prefix = "separate small context " * 100
                long_seed = send_retention_case("retention-long-seed", long_prefix + "seed")
                short_seed = send_retention_case("retention-short-seed", short_prefix + "seed")
                pressure[long_seed] = 0.35
                wait_observed(url)
                long_before = send_retention_case("retention-long-before", long_prefix + "before")
                short_before = send_retention_case("retention-short-before", short_prefix + "before")
                assert long_before == long_seed and short_before == short_seed
                started = time.monotonic()
                # Span a full thirty-second eviction interval on this fresh router.
                time.sleep(32)
                wait_observed(url)
                long_after = send_retention_case("retention-long-after", long_prefix + "after")
                short_after = send_retention_case("retention-short-after", short_prefix + "after")
                existing = send_retention_case("retention-long-seed", long_prefix + "bound continuation")
                retention = {
                    "scope": "Fresh released binary, mock backend; fixed pressure spans a full eviction interval.",
                    "elapsed_seconds": time.monotonic() - started,
                    "pressure": pressure.copy(),
                    "long_prefix_characters": len(long_prefix),
                    "short_prefix_characters": len(short_prefix),
                    "long_owner_before": long_before,
                    "long_owner_after": long_after,
                    "short_owner_before": short_before,
                    "short_owner_after": short_after,
                    "existing_session_owner_after": existing,
                    "requests": observed[extra_start:],
                }
                (retention_output / "result.json").write_text(json.dumps(retention, indent=2) + "\n")
        result = {
            "scope": (
                "CPU-only real released Router0.1.16 CLI with four mocked concrete DP ranks. No model, GPU, "
                "numerical correctness or throughput claim. Explicit mock token capacity is 1000000 per rank. Fresh "
                "observations verify equal KV pressure 0.2, then rank0 pressure 0.35 and 0.95. Tests prefix affinity, "
                "headroom fallback for new IDs, stable existing binding, JSON and SSE, preserved hash control and "
                "identical forwarded JSON bodies. The mock cannot establish live-model performance."
            ),
            "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "architecture": architecture,
            "results": results,
            "backend_http_counts": http_counts,
            "affinity_retention": retention,
            "status": "passed",
        }
        if args.test_affinity_retention:
            result["status"] = "diagnostic_completed"
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
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
