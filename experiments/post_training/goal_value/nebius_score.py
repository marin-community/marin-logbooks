# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score matched decision points and macro-average pass/fail ranking by issue."""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase

from experiments.post_training.goal_value.context import Context, ContextPolicy
from experiments.post_training.goal_value.model import (
    MODEL_ID,
    EncodedEpisode,
    TurnEncoder,
    ValueHead,
    decision_windows,
    load_checkpoint,
)
from experiments.post_training.goal_value.nebius import CUTOFFS, DATA_FILE, DATASET, REVISION, Scorable, prepare

TIE_TOLERANCE = 1e-6
BOOTSTRAP_REPLICATES = 10000
MAX_BATCH_SIZE = 16
PADDED_TOKEN_BUDGET = 32768


@dataclass(frozen=True)
class Window:
    encoded: EncodedEpisode
    cutoffs: tuple[int, ...]


def prefix_fingerprints(ids: np.ndarray, positions: list[int]) -> list[str]:
    """Identify equivalent inputs at causal decision positions."""
    return [hashlib.sha256(ids[: position + 1].tobytes()).hexdigest() for position in positions]


def windows_for(case: Scorable, tokenizer: PreTrainedTokenizerBase, context: Context) -> list[Window]:
    """Keep the checkpoint's pruning schedule and gather only prespecified decision states."""
    windows = []
    start = 0
    for window, unpruned in decision_windows(case.episode, tokenizer, context):
        end = start + len(window.positions)
        decisions = case.episode.predict_before[start:end]
        selected = [i for i, point in enumerate(decisions) if point in case.cutoffs]
        if selected:
            positions = [window.positions[i] for i in selected]
            encoded = replace(window, ids=window.ids[: positions[-1] + 1], positions=positions)
            windows.append(Window(encoded, tuple(case.cutoffs[decisions[i]] for i in selected)))
        if len(unpruned) != end - start:
            raise ValueError("bounded window lost a decision index")
        start = end
    if start != len(case.episode.predict_before) or sum(len(window.cutoffs) for window in windows) != len(case.cutoffs):
        raise ValueError("checkpoint context did not cover the requested cuts")
    return windows


@torch.inference_mode()
def score(cohort: list[Scorable], encoder: TurnEncoder, head: ValueHead, context: Context) -> tuple[list[dict], dict]:
    """Run length-sorted batches and retain only cutoff scores and their labels."""
    started = time.monotonic()
    windows = [window for case in cohort for window in windows_for(case, encoder.tokenizer, context)]
    windows.sort(key=lambda window: len(window.encoded.ids))
    print(f"Encoded {len(windows)} windows in {time.monotonic() - started:.1f}s", flush=True)

    start = 0
    padded_tokens = 0
    batches = 0
    records: dict[str, dict] = {}
    while start < len(windows):
        end = start + 1
        while end < min(len(windows), start + MAX_BATCH_SIZE):
            if len(windows[end].encoded.ids) * (end - start + 1) > PADDED_TOKEN_BUDGET:
                break
            end += 1
        batch = windows[start:end]
        results = encoder.hidden([window.encoded for window in batch])
        flattened = torch.cat([states for _, states in results]).to(device=encoder.device, dtype=torch.float32)
        values = iter(head(flattened).sigmoid().flatten().tolist())
        for window in batch:
            episode = window.encoded.episode
            row = records.setdefault(
                episode.trajectory_id,
                {
                    "trajectory_id": episode.trajectory_id,
                    "issue": episode.task_id,
                    "outcome": episode.outcome,
                    "scores": {},
                    "prefix_fingerprints": {},
                },
            )
            row["scores"].update({str(cut): next(values) for cut in window.cutoffs})
            fingerprints = prefix_fingerprints(np.asarray(window.encoded.ids, dtype=np.int32), window.encoded.positions)
            row["prefix_fingerprints"].update(zip(map(str, window.cutoffs), fingerprints, strict=True))
        padded_tokens += (end - start) * len(batch[-1].encoded.ids)
        start = end
        batches += 1
        if batches % 20 == 0:
            print(f"Scored {start}/{len(windows)} windows after {time.monotonic() - started:.1f}s", flush=True)
    if not all(row["scores"] for row in records.values()) or len(records) != len(cohort):
        raise ValueError("scoring coverage differs from the converted cohort")
    telemetry = {
        "windows": len(windows),
        "input_tokens": sum(len(window.encoded.ids) for window in windows),
        "padded_tokens": padded_tokens,
        "seconds": time.monotonic() - started,
    }
    return list(records.values()), telemetry


def issue_auc(rows: list[dict], cut: int) -> tuple[dict[str, float], dict]:
    """Compute one AUC per issue for a fixed cutoff."""
    groups = defaultdict(lambda: ([], []))
    key = str(cut)
    for row in rows:
        if key in row["scores"]:
            fingerprint = row.get("prefix_fingerprints", {}).get(key, "")
            groups[row["issue"]][row["outcome"]].append((row["scores"][key], fingerprint))
    scores = {}
    pairs = 0
    ties = 0
    identical_pairs = 0
    for issue, (negative, positive) in groups.items():
        if negative and positive:
            pos_values, pos_ids = zip(*positive, strict=True)
            neg_values, neg_ids = zip(*negative, strict=True)
            difference = np.asarray(pos_values)[:, None] - np.asarray(neg_values)
            pos_prefix = np.asarray(pos_ids)[:, None]
            neg_prefix = np.asarray(neg_ids)
            identical = (pos_prefix != "") & (pos_prefix == neg_prefix)
            tied = (np.abs(difference) <= TIE_TOLERANCE) | identical
            scores[issue] = float(np.mean(((difference > TIE_TOLERANCE) & ~tied) + 0.5 * tied))
            pairs += difference.size
            ties += int(tied.sum())
            identical_pairs += int(identical.sum())
    return scores, {
        "covered_attempts": sum(len(pass_scores) + len(fail_scores) for fail_scores, pass_scores in groups.values()),
        "covered_issues": len(groups),
        "mixed_issues": len(scores),
        "descriptive_pairs": pairs,
        "ties": ties,
        "identical_prefix_pairs": identical_pairs,
    }


def mean_and_interval(values: list[float]) -> dict:
    """Bootstrap issue-level means with a fixed seed."""
    if not values:
        return {"mean": None, "ci95": None}
    ties = np.asarray(values)
    draws = np.random.default_rng(79).choice(ties, size=(BOOTSTRAP_REPLICATES, len(ties)), replace=True)
    return {"mean": float(ties.mean()), "ci95": np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()}


def summarize(rows: list[dict]) -> dict:
    """Report conditional coverage and differences on the same [4, 8] attempt cohort."""
    by_cut = {}
    for cut in CUTOFFS:
        scores, coverage = issue_auc(rows, cut)
        by_cut[str(cut)] = {**coverage, **mean_and_interval(list(scores.values()))}
    comparable = [row for row in rows if {"4", "8"} <= row["scores"].keys()]
    four, _ = issue_auc(comparable, 4)
    eight, _ = issue_auc(comparable, 8)
    if four.keys() != eight.keys():
        raise ValueError("the matched comparison changed its issue population")
    return {
        "by_cutoff": by_cut,
        "matched_4_to_8": {
            "issues": len(four),
            "cut_4": mean_and_interval(list(four.values())),
            "cut_8": mean_and_interval(list(eight.values())),
            "delta": mean_and_interval([eight[issue] - score for issue, score in four.items()]),
        },
        "goal_only_within_issue": 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("excluded_issues", type=Path)
    parser.add_argument("--issues", type=int, default=192)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--model-dir", default=MODEL_ID)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("IRIS_OUTPUT_DIR", "results")))
    args = parser.parse_args()
    if args.issues < 1:
        parser.error("--issues must be positive")
    path = args.parquet or Path(hf_hub_download(DATASET, DATA_FILE, repo_type="dataset", revision=REVISION))
    excluded = set(args.excluded_issues.read_text().splitlines())
    cohort, audit = prepare(path, excluded, args.issues)
    print("NEBIUS_AUDIT " + json.dumps(audit), flush=True)
    encoder, head, context = load_checkpoint(args.checkpoint, args.model_dir)
    if context != Context(65536, ContextPolicy.PRUNE_MIDDLE):
        raise ValueError("expected the frozen 65,536-token middle-pruning checkpoint")
    rows, telemetry = score(cohort, encoder, head, context)
    metrics = {**summarize(rows), "audit": audit, "scoring": telemetry}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "nebius-predictions.jsonl").open("w") as handle:
        for row in sorted(rows, key=lambda row: row["trajectory_id"]):
            handle.write(json.dumps(row) + "\n")
    (args.output / "nebius-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("NEBIUS_METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
