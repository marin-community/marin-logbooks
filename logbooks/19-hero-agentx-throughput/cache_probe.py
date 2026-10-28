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


def post(base_url: str, payload: dict, rank: int) -> dict:
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
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


def run(tokenizer_path: Path, base_url: str, output: Path, prefix_tokens: int, rank: int) -> None:
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
    (output / "inputs.json").write_text(json.dumps({"tokenizer_sha256": TOKENIZER_SHA256, "prompts": prompts}))
    salt = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    # First obtain independent cold references. Then share one cache namespace
    # across different branches and varying API batch sizes, ending at branch A.
    waves = [
        ("cold-a", [0], salt + "-a"),
        ("cold-b", [1], salt + "-b"),
        ("cold-c", [2], salt + "-c"),
        ("cold-a-repeat", [0], salt + "-a-repeat"),
        ("shared-a", [0], salt + "-shared"),
        ("shared-ba", [1, 0], salt + "-shared"),
        ("shared-cab", [2, 0, 1], salt + "-shared"),
        ("shared-a-final", [0], salt + "-shared"),
    ]
    references = {}
    comparisons = []
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
            response = post(base_url, payload, rank)
            entry = {
                "wave": name,
                "branches": branches,
                "rank": rank,
                "started_unix": started,
                "elapsed_seconds": time.time() - started,
                "request": payload,
                "response": response,
            }
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            choices = sorted(response["choices"], key=lambda choice: choice["index"])
            assert len(choices) == len(branches)
            for branch, choice in zip(branches, choices, strict=True):
                if branch not in references:
                    references[branch] = choice
                else:
                    comparisons.append({"wave": name, "branch": branch, **compare(references[branch], choice)})
            print(name, "elapsed_seconds", round(entry["elapsed_seconds"], 3), "usage", response["usage"], flush=True)
    summary = {"rank": rank, "prefix_tokens": prefix_tokens, "comparisons": comparisons}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:38197")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    run(args.tokenizer, args.base_url, args.output, args.prefix_tokens, args.rank)


if __name__ == "__main__":
    main()
