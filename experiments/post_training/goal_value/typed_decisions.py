# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a typed decision model on held-out Snowball states."""

import argparse
import json
import math
import os
import time
from bisect import bisect_left
from pathlib import Path

import requests
from tokenizers import Tokenizer

from experiments.post_training.goal_value.data import Episode, read_shards, stable_hash
from experiments.post_training.goal_value.metrics import matched_turns, paired_difference, report, summarize

QUESTION = "Will this attempt pass the SWE-bench Verified grader when the agent submits its patch?"
OMITTED = "\n\n[Earlier actions and observations omitted to fit the context window.]\n"
SCHEMA_RESERVE = 512


def message_text(message: dict) -> str:
    """Render an archived turn as part of a document supplied to the classifier."""
    role = message["role"]
    fields = {key: value for key, value in message.items() if key != "role"}
    content = fields["content"]
    if not isinstance(content, str) or len(fields) > 1:
        content = json.dumps(fields, ensure_ascii=False, sort_keys=True)
    return f"\n\n[{role}]\n" + content.replace("<|", "<\\u007c")


def decision_state(episode: Episode, decision: int, tokenizer: Tokenizer, capacity: int) -> tuple[str, int, int]:
    """Keep the goal, initial task context, and the newest whole turns within the cap."""
    start = episode.predict_before[0]
    stop = episode.predict_before[decision]
    goal = episode.goal.replace("<|", "<\\u007c")
    protected = f"Goal and success criterion\n{goal}" + "".join(map(message_text, episode.messages[:start]))
    state = protected + "".join(map(message_text, episode.messages[start:stop]))
    encoded = tokenizer.encode(state, add_special_tokens=False)
    original = len(encoded.ids)
    if original <= capacity:
        return state, original, original

    minimum = protected + OMITTED
    tail_budget = capacity - len(tokenizer.encode(minimum, add_special_tokens=False).ids) - 16
    if tail_budget < 1 or decision == 0:
        raise ValueError(f"Initial context exceeds the state budget for {episode.trajectory_id}")
    earliest = encoded.offsets[original - tail_budget][0]
    boundaries = []
    offset = len(protected)
    turns = set(episode.predict_before[:decision])
    for index in range(start, stop):
        if index in turns:
            boundaries.append(offset)
        offset += len(message_text(episode.messages[index]))
    index = bisect_left(boundaries, earliest)
    tail = boundaries[index] if index < len(boundaries) else earliest
    state = minimum + state[tail:]
    retained = len(tokenizer.encode(state, add_special_tokens=False).ids)
    if retained > capacity:
        raise ValueError(f"Pruned state exceeds the state budget for {episode.trajectory_id}")
    return state, retained, original


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def evaluate(args: argparse.Namespace) -> None:
    """Score each requested decision once and persist scores as they arrive."""
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    matched = {row["trajectory_id"]: row for row in read_rows(args.baseline)}
    cohort = {episode.trajectory_id: episode for episode in read_shards(args.shards) if episode.split == "test"}
    if matched.keys() != cohort.keys():
        raise ValueError("Baseline trajectory IDs differ from the held-out data")
    predictions = args.output / "predictions.jsonl"
    completed = (
        {(row["trajectory_id"], row["decision"]) for row in read_rows(predictions) if row["model"] == args.model}
        if predictions.exists()
        else set()
    )
    queued = sorted(cohort, key=lambda identifier: stable_hash("clef-evaluation", identifier))
    with requests.Session() as client, predictions.open("a") as output:
        client.trust_env = False
        if args.token_env:
            client.headers["Authorization"] = "Bearer " + os.environ[args.token_env]
        for decision_name in args.decisions:
            for identifier in queued:
                if (identifier, decision_name) in completed:
                    continue
                episode = cohort[identifier]
                reference = matched[identifier]
                if episode.outcome != reference["outcome"]:
                    raise ValueError(f"Outcome does not match the baseline for {identifier}")
                decision = 0 if decision_name == "initial" else len(episode.predict_before) - 1
                state, retained, original = decision_state(
                    episode, decision, tokenizer, args.max_tokens - SCHEMA_RESERVE
                )
                if args.bucket is not None:
                    lower, upper = args.bucket
                    if not lower < retained + SCHEMA_RESERVE <= upper:
                        continue
                start = time.monotonic()
                response = client.post(
                    args.endpoint,
                    json={
                        "model": args.model,
                        "state": state,
                        "questions": {"passes": {"type": "noul", "instructions": QUESTION}},
                    },
                    timeout=600,
                )
                response.raise_for_status()
                answer = response.json()
                if answer["model"] != args.model:
                    raise ValueError(f"Endpoint served {answer['model']!r}, expected {args.model!r}")
                tokens = answer["usage"]["input_tokens"]
                probability = answer["answers"]["passes"]["noul"]
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(f"Invalid probability for {identifier}")
                if not retained <= tokens <= args.max_tokens:
                    raise ValueError(f"Server token count {tokens} is inconsistent with {retained} state tokens")
                row = {
                    "trajectory_id": identifier,
                    "task_id": episode.task_id,
                    "repository": episode.repository,
                    "family": episode.family,
                    "policy": episode.policy,
                    "outcome": episode.outcome,
                    "turn": decision + 1,
                    "total_turns": len(episode.predict_before),
                    "decision": decision_name,
                    "model": args.model,
                    "weight_format": args.weight_format,
                    "value": probability,
                    "reference": reference["head65_first" if decision_name == "initial" else "value"],
                    "task_only": reference["task_only"],
                    "global_rate": reference["global_rate"],
                    "family_rate": reference["family_rate"],
                    "input_tokens": tokens,
                    "server_context_tokens": args.bucket[1] if args.bucket is not None else args.max_tokens,
                    "state_tokens": retained,
                    "unpruned_state_tokens": original,
                    "seconds": time.monotonic() - start,
                }
                output.write(json.dumps(row) + "\n")
                output.flush()
                completed.add((identifier, decision_name))
                print(
                    f"{len(completed)} decisions saved, last {decision_name} used {tokens} tokens "
                    f"in {row['seconds']:.1f}s [P={probability:.4f}]",
                    flush=True,
                )


def analyze(output: Path, model: str, expected: int) -> None:
    """Compare each decision stage with the existing head on the same attempts."""
    rows = [row for row in read_rows(output / "predictions.jsonl") if row["model"] == model]
    grouped = {}
    policies = sorted({row["policy"] for row in rows})
    for policy in policies:
        for decision in ("initial", "final"):
            selected = [row for row in rows if row["decision"] == decision and row["policy"] == policy]
            if not selected:
                continue
            key = f"{policy}/{decision}"
            aligned = [dict(row, turn=1) for row in selected]
            grouped[key] = {
                "summary": summarize(selected),
                "reference": summarize([dict(row, value=row["reference"]) for row in selected])["value"],
                "paired_vs_reference": paired_difference(selected, "reference"),
                "paired_vs_global": paired_difference(selected, "global_rate"),
                "within_task": matched_turns(aligned),
                "pruned": sum(row["state_tokens"] < row["unpruned_state_tokens"] for row in selected),
                "tokens": sum(row["input_tokens"] for row in selected),
                "seconds": sum(row["seconds"] for row in selected),
            }
            if policy == policies[0]:
                report(selected, output / f"calibration-{decision}.png")
    (output / "metrics.json").write_text(
        json.dumps(
            {"model": model, "expected_predictions": expected, "predictions": len(rows), "results": grouped},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-env")
    parser.add_argument("--model", required=True)
    parser.add_argument("--weight-format", help="Record the checkpoint's weight format in each prediction")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--bucket", type=int, nargs=2, metavar=("LOWER_EXCLUSIVE", "UPPER_INCLUSIVE"))
    parser.add_argument("--decisions", choices=("initial", "final"), nargs="+", default=("final", "initial"))
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    if args.bucket is not None and not 0 <= args.bucket[0] < args.bucket[1] <= args.max_tokens:
        parser.error("--bucket requires 0 <= LOWER_EXCLUSIVE < UPPER_INCLUSIVE <= --max-tokens")
    args.output.mkdir(parents=True, exist_ok=True)
    evaluate(args)
    analyze(args.output, args.model, len(args.decisions) * len(read_rows(args.baseline)))


if __name__ == "__main__":
    main()
