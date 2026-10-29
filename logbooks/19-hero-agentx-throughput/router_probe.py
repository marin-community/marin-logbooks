# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check upstream routing against the native cache-owning engine and state controls."""

import argparse
import json
import os
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families
from router_process import upstream_router

from experiments.evaluation.agentx.processes import run_logged


def counters(url: str, output: Path) -> dict:
    with urllib.request.urlopen(url + "/metrics", timeout=30) as response:
        text = response.read().decode()
    output.write_text(text)
    names = {"vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"}
    return {
        (sample.name, int(sample.labels["engine"])): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-record", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    config = manifest["router"]
    server = json.loads(args.server_record.read_text())
    assert server["status"] == "ready"
    command = server["server_command"]
    prefix = command[: command.index("python")]
    environment = os.environ | {
        "UV_NO_CONFIG": "1",
        "UV_CONSTRAINT": "/app/experiments/evaluation/agentx/requirements-vllm-hero.txt",
    }
    probe = [*prefix, "python", str(Path(__file__).with_name("cache_probe.py")), "--tokenizer", str(args.tokenizer)]
    with upstream_router(config, args.output) as url:
        before = counters(config["backend_url"], args.output / "backend-metrics-before.txt")
        routed = args.output / "routed"
        run_logged(
            [*probe, "--base-url", url, "--session-id", config["session_id"], "--output", str(routed)],
            environment,
            args.output / "routed-probe.log",
            timeout=900,
        )
        after = counters(config["backend_url"], args.output / "backend-metrics-after.txt")
        deltas = [
            {"name": name, "engine": rank, "delta": value - before[(name, rank)]}
            for (name, rank), value in after.items()
        ]
        (args.output / "counter-deltas.json").write_text(json.dumps(deltas, indent=2) + "\n")
        owners = [
            row["engine"] for row in deltas if row["name"] == "vllm:prefix_cache_queries_total" and row["delta"] > 0
        ]
        assert len(owners) == 1, owners
        assert len(after) == 2 * config["data_parallel_size"]
        rank = owners[0]
        direct = args.output / "direct"
        run_logged(
            [*probe, "--base-url", config["backend_url"], "--rank", str(rank), "--output", str(direct)],
            environment,
            args.output / "direct-probe.log",
            timeout=900,
        )
        for directory in (direct, routed):
            comparisons = json.loads((directory / "comparison.json").read_text())["comparisons"]
            assert len(comparisons) == 8
            assert all(
                row["same_tokens"] and row["finite_logprobs"] and row["max_shared_logprob_absolute_delta"] == 0
                for row in comparisons
            )
        rows = []
        for a, b in zip(
            (direct / "requests.jsonl").read_text().splitlines(),
            (routed / "requests.jsonl").read_text().splitlines(),
            strict=True,
        ):
            left, right = json.loads(a), json.loads(b)
            assert left["wave"] == right["wave"] and left["branches"] == right["branches"]
            rows.append(
                {"wave": left["wave"], "exact_choices": left["response"]["choices"] == right["response"]["choices"]}
            )
        result = {
            "scope": "Functional DP-owner and numerical cache/history check; no AgentX throughput claim",
            "router": config,
            "selected_rank": rank,
            "comparisons": rows,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        assert len(rows) == 10 and all(row["exact_choices"] for row in rows)


if __name__ == "__main__":
    main()
