# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare cold, warm and interleaved native completions on one explicit DP rank.

This is a diagnostic with fixed short outputs, not an AgentX measurement. Keep
all token inputs and responses. Report differences without inventing a numerical
tolerance or assuming that an API session ID provides DP affinity.
"""

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_SHA256 = "881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2"
MODEL = "marin-community/Hero-535B-step144000"


def post(base_url: str, payload: dict, rank: int | None, session_id: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    if session_id is not None:
        headers["X-Session-ID"] = session_id
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def compare(reference: dict, observed: dict) -> dict:
    left = reference["logprobs"]
    right = observed["logprobs"]
    shared = []
    overlaps = []
    matching_prefix = 0
    for index, (a, b) in enumerate(zip(left["top_logprobs"], right["top_logprobs"], strict=True)):
        shared.extend(abs(a[token] - b[token]) for token in a.keys() & b.keys())
        overlaps.append(len(a.keys() & b.keys()))
        # After the first different token, later distributions have different
        # inputs and no longer measure the same conditional probabilities.
        if left["tokens"][index] != right["tokens"][index]:
            break
        matching_prefix += 1
    values = [value for row in right["top_logprobs"] for value in row.values()]
    return {
        "same_tokens": left["tokens"] == right["tokens"],
        "finite_logprobs": all(math.isfinite(value) for value in values),
        "max_shared_logprob_absolute_delta": max(shared, default=None),
        "shared_logprobs": len(shared),
        "matching_token_prefix": matching_prefix,
        "minimum_topk_overlap": min(overlaps),
    }


def batch_trace(base_url: str, method: str, args: list | None = None) -> dict:
    request = urllib.request.Request(
        base_url + "/collective_rpc",
        data=json.dumps({"method": method, "args": args or [], "kwargs": {}, "timeout": 60}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def run(
    tokenizer_path: Path,
    base_url: str,
    output: Path,
    prefix_tokens: int,
    rank: int | None,
    session_id: str | None,
    trace_batches: bool = False,
    reference_capture_sizes: list[int] | None = None,
    disable_graph_replay: bool = False,
    compare_to: Path | None = None,
    local_piecewise_dp: bool = False,
    moe_snapshot_rows: int | None = None,
) -> None:
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == TOKENIZER_SHA256
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    unit = tokenizer.encode(
        "The following notebook describes a parser. Each input is split into lines. "
        "A line has a name, an integer and an optional comment. Preserve line order. "
        "Reject invalid integers and keep the original text for error messages.\n",
        add_special_tokens=False,
    ).ids
    prefix = (unit * (prefix_tokens // len(unit) + 1))[:prefix_tokens]
    suffixes = (
        "\nExplain how to handle an empty input in three sentences.\nAnswer:",
        "\nExplain how to handle a negative integer in three sentences.\nAnswer:",
        "\nExplain how to handle a malformed comment in three sentences.\nAnswer:",
    )
    prompts = [prefix + tokenizer.encode(text, add_special_tokens=False).ids for text in suffixes]
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "tokenizer_sha256": TOKENIZER_SHA256,
                "prompts": prompts,
                "requested_dp_rank": rank,
                "trace_observer_dp_rank": 0 if trace_batches else None,
                "trace_scope": "First-engine RPC response contains shared DP vectors; every worker saves its own trace.",
            }
        )
    )
    salt = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    # First obtain independent cold references. Then share one cache namespace
    # across different branches and varying API batch sizes, ending at branch A.
    waves = [
        ("cold-a", [0], salt + "-a"),
        ("cold-b", [1], salt + "-b"),
        ("cold-c", [2], salt + "-c"),
        ("cold-a-repeat", [0], salt + "-a-repeat"),
        ("cold-ba", [1, 0], salt + "-ba"),
        ("cold-cab", [2, 0, 1], salt + "-cab"),
        ("shared-a", [0], salt + "-shared"),
        ("shared-ba", [1, 0], salt + "-shared"),
        ("shared-cab", [2, 0, 1], salt + "-shared"),
        ("shared-a-final", [0], salt + "-shared"),
    ]
    references = {}
    comparisons = []
    if trace_batches:
        assert rank is not None and rank >= 0
        batch_trace(
            base_url,
            "install_hero_batch_trace",
            [reference_capture_sizes, disable_graph_replay, local_piecewise_dp, True],
        )
        storage = batch_trace(base_url, "hero_cache_storage")
        (output / "cache-storage.json").write_text(json.dumps(storage, indent=2) + "\n")
        if moe_snapshot_rows is not None:
            assert disable_graph_replay and moe_snapshot_rows == 16
            installation = batch_trace(base_url, "install_hero_moe_snapshot", [48, moe_snapshot_rows])
            (output / "moe-snapshot-installation.json").write_text(json.dumps(installation, indent=2) + "\n")
    else:
        assert reference_capture_sizes is None and not disable_graph_replay and not local_piecewise_dp
        assert moe_snapshot_rows is None
    with (output / "requests.jsonl").open("w") as stream:
        for name, branches, cache_salt in waves:
            payload = {
                "model": MODEL,
                "prompt": [prompts[index] for index in branches],
                "max_tokens": 16,
                "min_tokens": 16,
                "ignore_eos": True,
                "temperature": 0,
                "seed": 42,
                "logprobs": 20,
                "return_tokens_as_token_ids": True,
                "cache_salt": cache_salt,
            }
            started = time.time()
            response = post(base_url, payload, rank, session_id)
            entry = {
                "wave": name,
                "branches": branches,
                "rank": rank,
                "session_id": session_id,
                "started_unix": started,
                "elapsed_seconds": time.time() - started,
                "request": payload,
                "response": response,
            }
            if trace_batches:
                entry["internal_batches"] = batch_trace(base_url, "take_hero_batch_trace")["results"]
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            choices = sorted(response["choices"], key=lambda choice: choice["index"])
            assert len(choices) == len(branches)
            for branch, choice in zip(branches, choices, strict=True):
                key = (tuple(branches), branch)
                if key not in references:
                    references[key] = (name, choice)
                else:
                    reference_name, reference = references[key]
                    comparisons.append(
                        {"wave": name, "branch": branch, "reference_wave": reference_name, **compare(reference, choice)}
                    )
            print(name, "elapsed_seconds", round(entry["elapsed_seconds"], 3), "usage", response["usage"], flush=True)
    if trace_batches:
        batch_trace(base_url, "stop_hero_batch_trace")
    summary = {"rank": rank, "session_id": session_id, "prefix_tokens": prefix_tokens, "comparisons": comparisons}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    if compare_to is not None:
        original = [json.loads(line) for line in (compare_to / "requests.jsonl").read_text().splitlines()]
        current = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
        cross = []
        for left, right in zip(original, current, strict=True):
            assert left["wave"] == right["wave"] and left["branches"] == right["branches"]
            shapes = []
            for entry in (left, right):
                shapes.append(
                    [
                        {
                            key: row[key]
                            for key in (
                                "requests",
                                "scheduled_tokens",
                                "padded_tokens",
                                "max_query_len",
                                "tokens_across_dp",
                            )
                        }
                        for row in entry["internal_batches"][0]
                    ]
                )
            cross.append(
                {
                    "wave": left["wave"],
                    "same_global_shapes": shapes[0] == shapes[1],
                    "choices": [
                        compare(a, b)
                        for a, b in zip(left["response"]["choices"], right["response"]["choices"], strict=True)
                    ],
                }
            )
        (output / "cross-comparison.json").write_text(json.dumps(cross, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:38197")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--trace-batches", action="store_true")
    parser.add_argument("--reference-capture-sizes", type=int, nargs="+")
    parser.add_argument("--disable-graph-replay", action="store_true")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--local-piecewise-dp", action="store_true")
    parser.add_argument("--moe-snapshot-rows", type=int)
    args = parser.parse_args()
    rank = args.rank if args.rank is not None or args.session_id is not None else 0
    run(
        args.tokenizer,
        args.base_url,
        args.output,
        args.prefix_tokens,
        rank,
        args.session_id,
        args.trace_batches,
        args.reference_capture_sizes,
        args.disable_graph_replay,
        args.compare_to,
        args.local_piecewise_dp,
        args.moe_snapshot_rows,
    )


if __name__ == "__main__":
    main()
