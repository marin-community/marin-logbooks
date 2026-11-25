# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare value-guided selection with the uniform mean of matched branch continuations."""

import argparse
import json
import random
from pathlib import Path
from statistics import mean, quantiles


def summarize(rows: list[dict], seeds: int, actions: int, tokens: int) -> dict:
    if len({row["issue"] for row in rows}) != len(rows):
        raise ValueError("expected one result per issue")
    complete = []
    for row in rows:
        if "replay_error" in row:
            continue
        pairs = list(row["continuations"].values())
        choice = row["guided_branch"]
        if len(pairs) != seeds or choice not in (0, 1, None):
            raise ValueError(f"unexpected seed or branch count for {row['issue']}")
        for pair in pairs:
            if len(pair) != 2 or any(
                branch["pass"] not in (0, 1)
                or not 1 <= branch["actor_calls"] <= actions
                or not 0 <= branch["generated_tokens"] <= tokens
                for branch in pair
            ):
                raise ValueError(f"invalid outcome or budget for {row['issue']}")
        guided = [pair[choice]["pass"] if choice is not None else mean(x["pass"] for x in pair) for pair in pairs]
        uniform = [mean(x["pass"] for x in pair) for pair in pairs]
        mixed = [len({pair[index]["pass"] for pair in pairs}) == 2 for index in (0, 1)]
        usage = {}
        for field in (
            "prompt_tokens",
            "generated_tokens",
            "actor_calls",
            "commands",
            "failed_commands",
            "malformed_actions",
        ):
            expected = sum(mean(x[field] for x in pair) for pair in pairs)
            selected = sum(pair[choice][field] for pair in pairs) if choice is not None else expected
            usage[field] = {"guided": selected, "uniform": expected}
        complete.append(
            {
                "issue": row["issue"],
                "value_scores": row["values"],
                "branch_passes": [sum(pair[index]["pass"] for pair in pairs) for index in (0, 1)],
                "guided_branch": choice,
                "guided_pass_rate": mean(guided),
                "uniform_pass_rate": mean(uniform),
                "guided_mixed_probability": float(mixed[choice]) if choice is not None else mean(mixed),
                "uniform_mixed_probability": mean(mixed),
                "discordant_seed_pairs": sum(pair[0]["pass"] != pair[1]["pass"] for pair in pairs),
                "usage": usage,
                "empty_patches": sum(branch["patch_bytes"] == 0 for pair in pairs for branch in pair),
                "explicit_submissions": sum(branch["explicit_submit"] for pair in pairs for branch in pair),
                "scoring_seconds": row["scoring_seconds"],
            }
        )
    result = {
        "issues_observed": len(rows),
        "issues_graded": len(complete),
        "excluded_replays": [row for row in rows if "replay_error" in row],
        "seeds_per_branch": seeds,
        "allocated_limits_per_continuation": {"actor_calls": actions, "generated_tokens": tokens},
        "issues": complete,
    }
    if complete:
        delta = [row["guided_pass_rate"] - row["uniform_pass_rate"] for row in complete]
        rng = random.Random(79)
        estimate = [mean(rng.choices(delta, k=len(delta))) for _ in range(4000)]
        intervals = quantiles(estimate, n=40, method="inclusive")
        result.update(
            guided_pass_rate=mean(row["guided_pass_rate"] for row in complete),
            uniform_pass_rate=mean(row["uniform_pass_rate"] for row in complete),
            paired_pass_delta=mean(delta),
            paired_issue_bootstrap95=[intervals[0], intervals[-1]] if any(delta) else None,
            issues_with_gain=sum(x > 0 for x in delta),
            issues_with_loss=sum(x < 0 for x in delta),
            guided_mixed_fraction=mean(row["guided_mixed_probability"] for row in complete),
            uniform_mixed_fraction=mean(row["uniform_mixed_probability"] for row in complete),
            discordant_seed_pairs=sum(row["discordant_seed_pairs"] for row in complete),
            empty_patches=sum(row["empty_patches"] for row in complete),
            explicit_submissions=sum(row["explicit_submissions"] for row in complete),
            total_scoring_seconds=sum(row["scoring_seconds"] for row in complete),
            usage={
                field: {arm: sum(row["usage"][field][arm] for row in complete) for arm in ("guided", "uniform")}
                for field in complete[0]["usage"]
            },
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--actions", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=6144)
    args = parser.parse_args()
    if min(args.seeds, args.actions, args.tokens) < 1:
        parser.error("continuation limits must be positive")
    rows = [json.loads(line) for line in args.scores.read_text().splitlines()]
    payload = json.dumps(summarize(rows, args.seeds, args.actions, args.tokens), indent=2) + "\n"
    args.output.write_text(payload)


if __name__ == "__main__":
    main()
