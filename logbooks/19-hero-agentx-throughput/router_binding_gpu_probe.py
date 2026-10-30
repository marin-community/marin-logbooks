# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify chat-prefix placement against cold references on the actual DP owner."""

import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path

from cache_probe import TOKENIZER_SHA256, batch_trace, compare
from router_probe import counters
from router_process import upstream_router
from tokenizers import Tokenizer

MODEL = "marin-community/Hero-535B-step144000"


def post(url: str, payload: dict, session_id: str, rank: int | None = None) -> dict:
    headers = {"Content-Type": "application/json", "X-Session-ID": session_id, "X-Request-ID": str(uuid.uuid4())}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def completion_logprobs(response: dict) -> dict:
    choices = response["choices"]
    assert len(choices) == 1
    content = choices[0]["logprobs"]["content"]
    assert len(content) == 16
    return {
        "logprobs": {
            "tokens": [row["token"] for row in content],
            "top_logprobs": [{entry["token"]: entry["logprob"] for entry in row["top_logprobs"]} for row in content],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    config = manifest["router"]
    assert config["program_binding"]["binding_strategy"] == "cache_aware"
    assert config["data_parallel_size"] == 16
    assert manifest["diagnostic_request"]["router_binding_probe"] is True
    assert hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() == TOKENIZER_SHA256
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    unit = tokenizer.encode(
        "The following notebook describes a parser. Each input is split into lines. "
        "A line has a name, an integer and an optional comment. Preserve line order. "
        "Reject invalid integers and keep the original text for error messages.\n",
        add_special_tokens=False,
    ).ids
    prefix = tokenizer.decode((unit * (8192 // len(unit) + 1))[:8192])
    salt = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()
    markers = {name: hashlib.sha256((salt + name).encode()).hexdigest() for name in ("alpha", "beta")}
    suffixes = ["Explain empty input.", "Explain negative integers.", "Explain malformed comments."]
    waves = [
        ("cold-a", "alpha", 0, "cold-a"),
        ("cold-b", "alpha", 1, "cold-b"),
        ("cold-c", "alpha", 2, "cold-c"),
        ("shared-a", "alpha", 0, "shared"),
        ("shared-b", "alpha", 1, "shared"),
        ("shared-c", "alpha", 2, "shared"),
        ("shared-a-final", "alpha", 0, "shared"),
        ("other-tree", "beta", 0, "shared"),
        ("other-tree-repeat", "beta", 0, "shared"),
        ("return-alpha", "alpha", 0, "shared"),
    ]
    storage = batch_trace(config["backend_url"], "hero_cache_storage")
    (args.output / "cache-storage.json").write_text(json.dumps(storage, indent=2) + "\n")
    rows = []
    with upstream_router(config, args.output) as url:
        for name, tree, branch, namespace in waves:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": f"[tree:{markers[tree]}]\n{prefix}\n{suffixes[branch]}"}],
                "max_tokens": 16,
                "min_tokens": 16,
                "ignore_eos": True,
                "temperature": 0,
                "seed": 42,
                "logprobs": True,
                "top_logprobs": 20,
                "return_tokens_as_token_ids": True,
                "stream": False,
                "cache_salt": salt + namespace,
            }
            before = counters(config["backend_url"], args.output / f"{name}-before.prom")
            started = time.time()
            response = post(url, payload, salt + name)
            elapsed = time.time() - started
            after = counters(config["backend_url"], args.output / f"{name}-after.prom")
            deltas = [
                {"metric": metric, "rank": rank, "delta": value - before[(metric, rank)]}
                for (metric, rank), value in after.items()
            ]
            assert len(after) == 32
            owners = [
                row["rank"] for row in deltas if row["metric"] == "vllm:prefix_cache_queries_total" and row["delta"] > 0
            ]
            assert len(owners) == 1, owners
            owner = owners[0]
            reference_payload = payload | {"cache_salt": salt + "-reference-" + name}
            reference = post(config["backend_url"], reference_payload, salt + "-reference-" + name, owner)
            comparison = compare(completion_logprobs(reference), completion_logprobs(response))
            assert comparison["same_tokens"] and comparison["finite_logprobs"]
            assert comparison["max_shared_logprob_absolute_delta"] == 0
            assert comparison["minimum_topk_overlap"] == 20 and comparison["matching_token_prefix"] == 16
            row = {
                "wave": name,
                "tree": tree,
                "branch": branch,
                "session_id": salt + name,
                "owner": owner,
                "elapsed_seconds": elapsed,
                "payload": payload,
                "response": response,
                "reference_payload": reference_payload,
                "reference_response": reference,
                "comparison": comparison,
                "counter_deltas": deltas,
                "hit_tokens": sum(x["delta"] for x in deltas if x["metric"] == "vllm:prefix_cache_hits_total"),
            }
            rows.append(row)
            (args.output / "requests.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(name, "owner", owner, "hit_tokens", row["hit_tokens"], flush=True)
        with urllib.request.urlopen(url + "/scheduling/diagnostics", timeout=10) as response:
            diagnostics = json.load(response)
        assert diagnostics["binding_only"] and diagnostics["retained_requests"] == 0
        (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    by_name = {row["wave"]: row for row in rows}
    alpha_owner = by_name["shared-a"]["owner"]
    for name in ("shared-b", "shared-c", "shared-a-final", "return-alpha"):
        assert by_name[name]["owner"] == alpha_owner
        assert by_name[name]["hit_tokens"] >= 8192
    assert by_name["shared-a"]["hit_tokens"] == 0
    assert by_name["other-tree"]["hit_tokens"] == 0
    assert by_name["other-tree-repeat"]["hit_tokens"] >= 8192
    result = {
        "status": "passed",
        "scope": (
            "Diagnostic fixed-output chat requests. Ten different conversation IDs, shared-prefix branches, "
            "two independently marked trees, and ten cold native references forced to each observed physical owner. "
            "No AgentX throughput claim. Logprobs compare fixed histories on the same live model and DP owner. "
            "All sixteen workers write native KV metadata; the RPC response alone represents the first engine."
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "router": config,
        "waves": [{k: row[k] for k in ("wave", "owner", "hit_tokens", "comparison")} for row in rows],
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ROUTER_BINDING_NATIVE_CONTROLS_PASSED", flush=True)


if __name__ == "__main__":
    main()
