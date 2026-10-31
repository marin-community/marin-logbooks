# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the released Router against two CPU backends with sixteen ranks each."""

import argparse
import json
import threading
import urllib.request
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from router_process import upstream_router

MODEL = "marin-community/Hero-535B-step144000"


def backend(group: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, data: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/metrics":
                rows = ["# TYPE vllm:num_requests_running gauge"]
                rows += [f'vllm:num_requests_running{{engine="{rank}",model_name="{MODEL}"}} 0' for rank in range(16)]
                rows += [f'probe_backend_identity{{group="{group}"}} 1']
                self.reply(("\n".join(rows) + "\n").encode(), "text/plain; version=0.0.4")
                return
            self.reply(json.dumps({"model_id": MODEL, "data": [{"id": MODEL}]}).encode(), "application/json")

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = {
                "group": group,
                "rank": self.headers.get("X-data-parallel-rank"),
                "session": self.headers.get("X-Session-ID"),
            }
            self.reply(json.dumps(body).encode(), "application/json")

        def log_message(self, *args) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--router-port", type=int, default=38450)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.manifest.read_text())["router"].copy()
    config.pop("backend_url", None)
    config["base_url"] = f"http://127.0.0.1:{args.router_port}"
    config["data_parallel_size"] = 16
    config["policy"] = "consistent_hash"
    config.pop("program_binding", None)
    with ExitStack() as stack:
        servers = [backend(group) for group in range(2)]
        for server in servers:
            stack.callback(server.server_close)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(server.shutdown)
        config["backend_urls"] = [f"http://127.0.0.1:{server.server_port}" for server in servers]
        assignments = {}
        with upstream_router(config, args.output) as url:
            for iteration in range(2):
                for index in range(128):
                    session = f"hero-replicas-wire-{index}"
                    request = urllib.request.Request(
                        url + "/v1/completions",
                        data=json.dumps({"model": MODEL, "prompt": "test", "max_tokens": 1}).encode(),
                        headers={"Content-Type": "application/json", "X-Session-ID": session},
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        row = json.load(response)
                    owner = [row["group"], int(row["rank"])]
                    assert owner[0] in (0, 1) and 0 <= owner[1] < 16 and row["session"] == session
                    if iteration == 0:
                        assignments[session] = owner
                    else:
                        assert assignments[session] == owner
            assert {owner[0] for owner in assignments.values()} == {0, 1}
            direct = []
            for group, base in enumerate(config["backend_urls"]):
                with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
                    text = response.read().decode()
                assert f'probe_backend_identity{{group="{group}"}} 1' in text
                assert sum(line.startswith("vllm:num_requests_running{") for line in text.splitlines()) == 16
                (args.output / f"backend-{group}-metrics.txt").write_text(text)
                direct.append({"group": group, "endpoint": base + "/metrics", "engines": 16})
            proxied_groups = []
            for iteration in range(8):
                with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
                    text = response.read().decode()
                groups = [group for group in range(2) if f'probe_backend_identity{{group="{group}"}} 1' in text]
                assert len(groups) == 1
                proxied_groups.extend(groups)
                (args.output / f"router-metrics-{iteration}.txt").write_text(text)
            result = {
                "requests": 256,
                "backends": 2,
                "dp_per_backend": 16,
                "stable_assignments": assignments,
                "observed_owners": sorted({tuple(owner) for owner in assignments.values()}),
                "direct_metrics": direct,
                "router_metrics_selected_groups": proxied_groups,
                "scope": (
                    "Actual released Router with CPU mock backends. Stable ownership and rank headers only; "
                    "no model, network-between-hosts or performance qualification. Router metrics proxy selects one "
                    "backend per scrape. Future multi-replica accounting must use both direct endpoints once each "
                    "and exclude the automatic Router alias from server totals."
                ),
            }
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: result[key] for key in ["requests", "backends", "dp_per_backend", "scope"]}))


if __name__ == "__main__":
    main()
