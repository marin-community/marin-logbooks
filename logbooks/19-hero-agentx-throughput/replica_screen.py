# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen one native DP32 group or two independent native DP16 groups.

Run on the first group's actual leader after both servers reach readiness.
The submitting process must fence every native task before and after this
client, retain the raw results, and stop both groups on any failed attempt.
Client completion alone never qualifies a deployment or an AgentX result.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from flag_controls import prepare_tokenizer
from iris.cluster.client.job_info import get_job_info
from prometheus_client.parser import text_string_to_metric_families
from router_process import upstream_router

from experiments.evaluation.agentx.processes import run_logged


def metrics(backends: list[dict], output: Path) -> dict:
    """Preserve endpoint identity when independent groups reuse engine IDs."""
    result = {}
    for index, backend in enumerate(backends):
        endpoint = backend["url"] + "/metrics"
        with urllib.request.urlopen(endpoint, timeout=30) as response:
            text = response.read().decode()
        (output / f"backend-{index}.prom").write_text(text)
        rows = {str(rank): {} for rank in range(backend["dp"])}
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                rank = sample.labels.get("engine")
                if rank not in rows:
                    continue
                if sample.name in {
                    "vllm:request_success_total",
                    "vllm:prefix_cache_queries_total",
                    "vllm:prefix_cache_hits_total",
                    "vllm:prompt_tokens_total",
                    "vllm:generation_tokens_total",
                    "vllm:num_preemptions_total",
                    "vllm:kv_cache_usage_perc",
                }:
                    rows[rank][sample.name] = rows[rank].get(sample.name, 0) + sample.value
        assert all("vllm:kv_cache_usage_perc" in row for row in rows.values())
        result[endpoint] = rows
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def successful_requests(snapshot: dict) -> dict[tuple[str, str], float]:
    return {
        (endpoint, rank): row.get("vllm:request_success_total", 0)
        for endpoint, ranks in snapshot.items()
        for rank, row in ranks.items()
    }


def completed_probe_owner(backends: list[dict], before: dict, output: Path) -> tuple[str, str]:
    output.mkdir()
    deadline = time.monotonic() + 30
    while True:
        after = successful_requests(metrics(backends, output))
        delta = {key: value - before[key] for key, value in after.items()}
        assert all(value >= 0 for value in delta.values()), delta
        if sum(delta.values()) >= 16:
            break
        assert time.monotonic() < deadline, "Request counters did not catch up with the completed probe"
        time.sleep(0.25)
    active = [key for key, value in delta.items() if value]
    assert len(active) == 1 and delta[active[0]] == 16, delta
    return active[0]


def check_probe(output: Path) -> None:
    rows = json.loads((output / "comparison.json").read_text())["comparisons"]
    assert len(rows) == 8 and all(
        row["same_tokens"]
        and row["finite_logprobs"]
        and row["max_shared_logprob_absolute_delta"] == 0
        and row["minimum_topk_overlap"] == 20
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    backends = manifest["replica_backends"]
    assert [backend["dp"] for backend in backends] in ([32], [16, 16])
    assert len({backend["url"] for backend in backends}) == len(backends)
    info = get_job_info()
    assert info is not None and info.task_index == 0
    assert str(info.job_id) == backends[0]["job_id"]
    assert info.num_tasks * 4 == backends[0]["dp"]
    primary = json.loads(args.primary_diagnostic.read_text())
    assert primary["status"] == "ready" and primary["api_bind"]["url"] == backends[0]["url"]
    assert primary["gang_attempts"] == backends[0]["gang_attempts"]
    command = primary["server_command"]
    prefix = [
        "env",
        "UV_NO_CONFIG=1",
        "UV_CONSTRAINT=/app/experiments/evaluation/agentx/requirements-vllm-hero.txt",
        *command[: command.index("python")],
    ]
    assert "uvx" in prefix and "--from" in prefix
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(args.manifest.read_text())
    tokenizer = prepare_tokenizer(manifest, args.output / "tokenizer")
    probe = [
        *prefix,
        "python",
        str(Path(__file__).with_name("cache_probe.py")),
        "--tokenizer",
        str(tokenizer / "tokenizer.json"),
    ]
    for index, backend in enumerate(backends):
        before_dir = args.output / f"direct-{index}-before"
        before_dir.mkdir()
        before = successful_requests(metrics(backends, before_dir))
        output = args.output / f"direct-state-{index}"
        run_logged(
            [*probe, "--base-url", backend["url"], "--rank", "0", "--output", str(output)],
            dict(os.environ),
            args.output / f"direct-state-{index}.log",
            timeout=900,
        )
        check_probe(output)
        owner = completed_probe_owner(backends, before, args.output / f"direct-{index}-after")
        assert owner == (backend["url"] + "/metrics", "0"), owner
    router = manifest["router"] | {
        "backend_urls": [backend["url"] for backend in backends],
        "data_parallel_size": backends[0]["dp"],
    }
    router.pop("backend_url", None)
    owners = []
    with upstream_router(router, args.output) as base_url:
        for index in range(16):
            before_dir = args.output / f"route-{index}-before"
            before_dir.mkdir()
            before = successful_requests(metrics(backends, before_dir))
            output = args.output / f"routed-state-{index}"
            session = f"hero-replica-state-{index}"
            run_logged(
                [*probe, "--base-url", base_url, "--session-id", session, "--output", str(output)],
                dict(os.environ),
                args.output / f"routed-state-{index}.log",
                timeout=900,
            )
            check_probe(output)
            owner = completed_probe_owner(backends, before, args.output / f"route-{index}-after")
            owners.append({"session": session, "endpoint": owner[0], "engine": owner[1]})
            if {owner["endpoint"] for owner in owners} == {backend["url"] + "/metrics" for backend in backends}:
                break
        else:
            raise RuntimeError("Fixed session controls did not exercise every backend")
        (args.output / "routed-state-owners.json").write_text(json.dumps(owners, indent=2) + "\n")
        for backend in backends:
            request = urllib.request.Request(backend["url"] + "/reset_prefix_cache", data=b"", method="POST")
            with urllib.request.urlopen(request, timeout=90) as response:
                assert json.load(response) == {"success": True}
        before_dir = args.output / "screen-before"
        before_dir.mkdir()
        metrics(backends, before_dir)
        for case in manifest["screen_cases"]:
            output = args.output / "clients" / case["name"]
            output.mkdir(parents=True)
            argv = [
                *prefix,
                *case["argv"],
                "--result-dir",
                str(output),
                "--result-filename",
                "result.json",
            ]
            argv[argv.index("--base-url") + 1] = base_url
            argv[argv.index("--tokenizer") + 1] = str(tokenizer)
            (output / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
            run_logged(argv, dict(os.environ), output / "client.log", timeout=case["timeout_seconds"])
            result = json.loads((output / "result.json").read_text())
            assert result["failed"] == 0 and result["completed"] == int(argv[argv.index("--num-prompts") + 1])
        after_dir = args.output / "screen-after"
        after_dir.mkdir()
        metrics(backends, after_dir)
    record = {
        "status": "client_completed_pending_external_native_attempt_fences_and_retention",
        "routed_state_owners": owners,
        "physical_gpus": 32,
        "scope": (
            "Synthetic client only. No AgentX result. Preserve each endpoint/engine identity; no Router metrics alias."
        ),
    }
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
