# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect the pinned model's native routed-expert IDs on prepared long histories.

Capture adds work to the server. Counts describe routing frequency, not measured
throughput or the critical path. An alternate grouping is a load hypothesis.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

from experiments.evaluation.agentx.artifacts import copy_file

MODEL = "marin-community/Hero-535B-step144000"


def post(path: str, body: dict, rank: int) -> dict:
    request = urllib.request.Request(
        "http://127.0.0.1:38197" + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = manifest["diagnostic_request"]["expert_load"]
    inputs = args.output / "expert-prompts.json"
    copy_file(source["uri"], str(inputs))
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == source["sha256"]
    prompts = json.loads(inputs.read_text())

    def collect(item: tuple[int, dict]) -> dict:
        rank, sample = item
        tokens = post("/tokenize", {"model": MODEL, "messages": sample["messages"]}, rank)["tokens"]
        assert 0 < len(tokens) < 262144 - 128
        body = {
            "model": MODEL,
            "prompt": tokens,
            "max_tokens": 128,
            "min_tokens": 128,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 42,
            "logprobs": 5,
            "return_tokens_as_token_ids": True,
            "routed_experts_prompt_start": len(tokens),
        }
        (args.output / f"expert-request-{rank}.json").write_text(json.dumps(body) + "\n")
        response = post("/v1/completions", body, rank)
        (args.output / f"expert-response-{rank}.json").write_text(json.dumps(response) + "\n")
        choice = response["choices"][0]
        assert all(math.isfinite(value) for row in choice["logprobs"]["top_logprobs"] for value in row.values())
        routes = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])), allow_pickle=False)
        assert routes.ndim == 3 and routes.shape[1:] == (48, 8), routes.shape
        assert 0 < routes.shape[0] <= 128 and np.issubdtype(routes.dtype, np.integer)
        assert routes.min() >= 0 and routes.max() < 384
        np.save(args.output / f"expert-routes-{rank}.npy", routes, allow_pickle=False)
        counts = np.stack([np.bincount(routes[:, layer, :].reshape(-1), minlength=384) for layer in range(48)])
        return {
            "rank": rank,
            "source_trace_id": sample["source_trace_id"],
            "prompt_tokens": len(tokens),
            "completion_tokens": response["usage"]["completion_tokens"],
            "routed_shape": list(routes.shape),
            "counts": counts.tolist(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        requests = list(executor.map(collect, enumerate(prompts["samples"])))
    counts = np.asarray([row.pop("counts") for row in requests], dtype=np.int64).sum(axis=0)
    linear = counts.reshape(48, 16, 24).sum(axis=2)
    alternate = counts.reshape(48, 24, 16).sum(axis=1)
    summary = {
        "scope": "Intrusive native routing capture on16 long first turns, not AgentX confirmation or throughput.",
        "input": source,
        "requests": requests,
        "expert_counts_by_layer": counts.tolist(),
        "native_linear_rank_counts_by_layer": linear.tolist(),
        "hypothetical_round_robin_counts_by_layer": alternate.tolist(),
        "linear_max_mean_by_layer": (linear.max(axis=1) / linear.mean(axis=1)).tolist(),
        "hypothetical_round_robin_max_mean_by_layer": (alternate.max(axis=1) / alternate.mean(axis=1)).tolist(),
    }
    (args.output / "expert-load.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
