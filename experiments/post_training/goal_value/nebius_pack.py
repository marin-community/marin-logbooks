# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack or join the same decision windows and head for TPU scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.post_training.goal_value.context import Context, ContextPolicy
from experiments.post_training.goal_value.model import ENCODING_VERSION, MODEL_ID, MODEL_REVISION
from experiments.post_training.goal_value.nebius import prepare
from experiments.post_training.goal_value.nebius_score import prefix_fingerprints, summarize, windows_for


def pack(args: argparse.Namespace) -> None:
    """Write bounded token windows and the frozen affine readout."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint[name] for name in ("model", "revision", "encoding_version")) != (
        MODEL_ID,
        MODEL_REVISION,
        ENCODING_VERSION,
    ):
        raise ValueError("unexpected value-head checkpoint")
    context = Context(checkpoint["max_tokens"], ContextPolicy(checkpoint["context_policy"]))
    if context != Context(65536, ContextPolicy.PRUNE_MIDDLE):
        raise ValueError("unexpected value-head context")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    excluded = set(args.excluded.read_text().splitlines())
    cohort, audit = prepare(args.parquet, excluded, args.issues)
    print(f"Packing {len(cohort)} converted attempts", flush=True)
    tokens = []
    takes = []
    layout = []
    for case in cohort:
        for window in windows_for(case, tokenizer, context):
            tokens.append(np.asarray(window.encoded.ids, dtype=np.int32))
            takes.append(np.asarray(window.encoded.positions, dtype=np.int32))
            layout.append(
                {
                    "trajectory_id": case.episode.trajectory_id,
                    "issue": case.episode.task_id,
                    "outcome": case.episode.outcome,
                    "cutoffs": window.cutoffs,
                }
            )
    np.savez_compressed(
        args.output / "nebius-inputs.npz",
        ids=np.concatenate(tokens),
        indices=np.concatenate(takes),
        id_offsets=np.cumsum([0, *map(len, tokens)]),
        index_offsets=np.cumsum([0, *map(len, takes)]),
        weight=checkpoint["state_dict"]["weight"].float().numpy().ravel(),
        bias=checkpoint["state_dict"]["bias"].float().numpy(),
    )
    (args.output / "nebius-layout.json").write_text(json.dumps({"audit": audit, "windows": layout}) + "\n")
    print(f"Packed {len(tokens)} windows with {sum(map(len, tokens)):,} input tokens", flush=True)


def join(args: argparse.Namespace) -> None:
    """Join TPU values to the identical cohort and compute issue-level metrics."""
    manifest = json.loads(args.layout.read_text())
    values = np.load(args.values)
    records = {}
    with np.load(args.packed_inputs) as payload:
        ids, id_offsets = payload["ids"], payload["id_offsets"]
        positions, index_offsets = payload["indices"], payload["index_offsets"]
    for index, (entry, window_scores) in enumerate(zip(manifest["windows"], values, strict=True)):
        row = records.setdefault(
            entry["trajectory_id"],
            {k: v for k, v in entry.items() if k != "cutoffs"} | {"scores": {}, "prefix_fingerprints": {}},
        )
        window_ids = ids[id_offsets[index] : id_offsets[index + 1]]
        selected = positions[index_offsets[index] : index_offsets[index + 1]].tolist()
        row["prefix_fingerprints"].update(
            zip(map(str, entry["cutoffs"]), prefix_fingerprints(window_ids, selected), strict=True)
        )
        for cut, score in zip(entry["cutoffs"], window_scores[: len(entry["cutoffs"])], strict=True):
            if str(cut) in row["scores"] or not np.isfinite(score):
                raise ValueError("TPU results missed or duplicated a cutoff")
            row["scores"][str(cut)] = float(score)
    rows = list(records.values())
    metrics = {**summarize(rows), "audit": manifest["audit"], "scoring": json.loads(args.telemetry.read_text())}
    (args.output / "nebius-predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.output / "nebius-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("NEBIUS_METRICS " + json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    packer = modes.add_parser("pack")
    for field in ("parquet", "checkpoint", "excluded", "output"):
        packer.add_argument(field, type=Path)
    packer.add_argument("--issues", type=int, default=192)
    packer.set_defaults(func=pack)
    joiner = modes.add_parser("join")
    for field in ("layout", "packed_inputs", "values", "telemetry", "output"):
        joiner.add_argument(field, type=Path)
    joiner.set_defaults(func=join)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
