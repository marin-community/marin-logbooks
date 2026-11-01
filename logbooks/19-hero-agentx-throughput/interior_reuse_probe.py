# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure first reuse of an interior prefix after a longer prompt completes."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from cache_probe import MODEL, TOKENIZER_SHA256, compare, post
from tokenizers import Tokenizer


def run(tokenizer_path: Path, output: Path) -> None:
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == TOKENIZER_SHA256
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    unit = tokenizer.encode(
        "Each input line contains a name, an integer and an optional comment. "
        "Preserve line order and reject invalid integers.\n",
        add_special_tokens=False,
    ).ids
    history = (unit * (49152 // len(unit) + 1))[:49152]
    branch = (
        history[:24576]
        + tokenizer.encode(
            "\nExplain how to handle an empty input in three sentences.\nAnswer:", add_special_tokens=False
        ).ids
    )
    longer = (
        history
        + tokenizer.encode(
            "\nExplain how to handle a negative integer in three sentences.\nAnswer:", add_special_tokens=False
        ).ids
    )
    salt = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    waves = [
        ("cold-branch", branch, salt + "-cold"),
        ("cold-branch-repeat", branch, salt + "-cold-repeat"),
        ("warm-long", longer, salt + "-shared"),
        ("first-interior-branch", branch, salt + "-shared"),
        ("repeat-interior-branch", branch, salt + "-shared"),
    ]
    rows = []
    reference = None
    with (output / "requests.jsonl").open("w") as stream:
        for name, prompt, namespace in waves:
            payload = {
                "model": MODEL,
                "prompt": [prompt],
                "max_tokens": 16,
                "min_tokens": 16,
                "ignore_eos": True,
                "temperature": 0,
                "seed": 42,
                "logprobs": 20,
                "return_tokens_as_token_ids": True,
                "cache_salt": namespace,
            }
            started = time.time()
            response = post("http://127.0.0.1:38197", payload, rank=0, session_id=None)
            elapsed = time.time() - started
            stream.write(
                json.dumps(
                    {
                        "wave": name,
                        "started_unix": started,
                        "elapsed_seconds": elapsed,
                        "request": payload,
                        "response": response,
                    }
                )
                + "\n"
            )
            stream.flush()
            assert len(response["choices"]) == 1
            choice = response["choices"][0]
            if reference is None:
                reference = choice
            check = None if name in ("cold-branch", "warm-long") else compare(reference, choice)
            usage = response["usage"]
            row = {
                "wave": name,
                "elapsed_seconds": elapsed,
                "prompt_tokens": usage["prompt_tokens"],
                "cache_read_tokens": usage["prompt_tokens_details"]["cached_tokens"],
                "comparison_to_cold_branch": check,
            }
            rows.append(row)
            (output / "result.json").write_text(json.dumps({"status": "running", "waves": rows}, indent=2) + "\n")
            if check is not None:
                assert check["same_tokens"] and check["finite_logprobs"]
                assert check["max_shared_logprob_absolute_delta"] == 0
                assert check["minimum_topk_overlap"] == 20 and check["matching_token_prefix"] == 16
    result = {
        "status": "all_interior_branch_state_checks_exact",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "history_tokens": 49152,
        "interior_history_tokens": 24576,
        "requested_dp_rank": 0,
        "waves": rows,
        "scope": (
            "Five sequential native completions with distinct cold and shared cache salts. "
            "All branch outputs pass the unchanged exact token and top-20 logprob gates. "
            "The first interior branch can reuse checkpoints created by the longer prompt. "
            "API cache-read tokens and single-request durations describe this synthetic reuse case; "
            "they do not establish AgentX throughput. Complete token inputs and parsed responses are retained."
        ),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.tokenizer, args.output)


if __name__ == "__main__":
    main()
