# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit and evaluate a linear BCE value head on frozen causal decision states."""

import argparse
import copy
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.post_training.goal_value.context import Context, ContextPolicy
from experiments.post_training.goal_value.data import read_shards, stable_hash
from experiments.post_training.goal_value.metrics import report
from experiments.post_training.goal_value.model import (
    ENCODING_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    TurnEncoder,
    ValueHead,
    cache_key,
    decision_windows,
    extract,
    feature_rows,
)


@dataclass(frozen=True)
class FitConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int


@dataclass(frozen=True)
class Corpus:
    states: torch.Tensor
    goal_states: torch.Tensor
    outcomes: torch.Tensor
    rows: list[dict]
    coverage: dict
    features: list[dict]

    def indices(self, split: str, policy: str) -> list[int]:
        return [i for i, row in enumerate(self.rows) if row["split"] == split and row["policy"] == policy]


def read_corpus(paths: list[Path], context: Context, batch_size: int, cache_dir: Path, model_dir: str) -> Corpus:
    """Reuse cached states for old shards and only encode shards added to the input."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    encoder = None
    seen = set()
    rows = []
    tensors = []
    goal_tensors = []
    goal_indices = {}
    coverage = defaultdict(Counter)
    feature_stats = []
    for path in paths:
        key = cache_key(path, context)
        cached = cache_dir / f"{key}.pt"
        if not cached.exists():
            if encoder is None:
                encoder = TurnEncoder(model_dir, device)
            cache = extract(encoder, path, context, batch_size)
            with tempfile.TemporaryDirectory(dir=cache_dir) as stage:
                staged = Path(stage) / cached.name
                torch.save(cache, staged)
                os.replace(staged, cached)
        else:
            cache = torch.load(cached, map_location="cpu", weights_only=True)
        feature_stats.append(
            {
                "shard": str(path),
                "key": key,
                "seconds": cache["seconds"],
                "input_tokens": sum(row["processed_tokens"] for row in cache["metadata"]),
            }
        )
        tensors.append(cache["states"])
        for row, goal_state in zip(cache["metadata"], cache["goal_states"], strict=True):
            identity = row["trajectory_id"]
            if identity in seen:
                raise ValueError(f"duplicate trajectory across shards: {identity}")
            seen.add(identity)
            covered = row["positions"]
            original = row["source_positions"]
            pruned_turns = sum(a != b for a, b in zip(covered, original, strict=True))
            for key in (row["split"], f"{row['split']}/{row['policy']}"):
                census = coverage[key]
                census["traces"] += 1
                census["traces_with_predictions"] += bool(covered)
                census["total_turns"] += row["total_turns"]
                census["covered_turns"] += len(covered)
                census["traces_with_all_turns"] += len(covered) == row["total_turns"]
                census["pruned_turns"] += pruned_turns
            if not covered:
                continue
            goal_key = (row["task_id"], row["goal_id"])
            if goal_key not in goal_indices:
                goal_indices[goal_key] = len(goal_tensors)
                goal_tensors.append(goal_state)
            metadata = {
                key: value
                for key, value in row.items()
                if key not in ("positions", "source_positions", "processed_tokens", "goal_id")
            }
            for turn, (position, source_position) in enumerate(zip(covered, original, strict=True)):
                rows.append(
                    dict(
                        metadata,
                        turn=turn,
                        input_tokens=position + 1,
                        unpruned_input_tokens=source_position + 1,
                        goal_index=goal_indices[goal_key],
                    )
                )
    if not rows:
        raise ValueError("no decision fits the configured context length")
    return Corpus(
        torch.cat(tensors).float().to(device),
        torch.stack(goal_tensors).float().to(device)[[row["goal_index"] for row in rows]],
        torch.tensor([row["outcome"] for row in rows], dtype=torch.float32, device=device),
        rows,
        {key: dict(value) for key, value in coverage.items()},
        feature_stats,
    )


def logits(head: ValueHead, states: torch.Tensor) -> torch.Tensor:
    return head(states).squeeze(-1)


def fit(
    corpus: Corpus,
    train_rows: list[int],
    validation_rows: list[int],
    config: FitConfig,
    initial_head: ValueHead | None = None,
) -> tuple[ValueHead, dict]:
    """Optimize the mean BCE across supervised decision states."""
    if not train_rows or not validation_rows:
        raise ValueError("training and validation each require covered decisions")
    torch.manual_seed(config.seed)
    device = corpus.states.device
    head = copy.deepcopy(initial_head) if initial_head else ValueHead(corpus.states.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=0.01)
    train_ids = torch.tensor(train_rows, device=device)
    validation_ids = torch.tensor(validation_rows, device=device)
    rng = torch.Generator(device=device).manual_seed(config.seed)
    best_loss = float("inf")
    best = {}
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        shuffled = train_ids[torch.randperm(len(train_ids), device=device, generator=rng)]
        for batch in shuffled.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(logits(head, corpus.states[batch]), corpus.outcomes[batch])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss = F.binary_cross_entropy_with_logits(
                logits(head, corpus.states[validation_ids]), corpus.outcomes[validation_ids]
            ).item()
        if loss < best_loss:
            best_loss = loss
            best = {"epoch": epoch, "state": copy.deepcopy(head.state_dict())}
    head.load_state_dict(best["state"])
    seconds = time.perf_counter() - started
    return head.eval(), {
        "best_epoch": best["epoch"],
        "validation_log_loss": best_loss,
        "seconds": seconds,
        "turns_per_second": config.epochs * len(train_rows) / seconds,
        "training_turns": len(train_rows),
        "validation_turns": len(validation_rows),
    }


@torch.no_grad()
def predict_rows(
    corpus: Corpus,
    selected: list[int],
    head: ValueHead,
    task_head: ValueHead,
    global_rate: float,
    family_rates: dict,
) -> list[dict]:
    if not selected:
        return []
    selected_rows = [corpus.rows[i] for i in selected]
    full = logits(head, corpus.states[selected]).sigmoid().tolist()
    task = logits(task_head, corpus.goal_states[selected]).sigmoid().tolist()
    return [
        dict(
            row,
            value=value,
            task_only=task_only,
            global_rate=global_rate,
            family_rate=family_rates.get(row["family"], global_rate),
        )
        for row, value, task_only in zip(selected_rows, full, task, strict=True)
    ]


def initial_turns(corpus: Corpus, selected: list[int]) -> list[int]:
    return [i for i in selected if corpus.rows[i]["turn"] == 0]


def save_checkpoint(path: Path, head: ValueHead, context: Context) -> None:
    torch.save(
        {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "encoding_version": ENCODING_VERSION,
            "max_tokens": context.max_tokens,
            "context_policy": context.policy.value,
            "state_dict": {key: tensor.cpu() for key, tensor in head.state_dict().items()},
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    context = Context(args.max_tokens, ContextPolicy(args.context_policy))
    corpus = read_corpus(args.shards, context, args.encode_batch_size, args.cache_dir, args.model_dir)
    train_rows = corpus.indices("train", args.train_policy)
    validation_rows = corpus.indices("validation", args.train_policy)
    config = FitConfig(epochs=args.epochs, batch_size=args.head_batch_size, learning_rate=args.lr, seed=17)
    head, fit_stats = fit(corpus, train_rows, validation_rows, config)
    train_initial = initial_turns(corpus, train_rows)
    task_corpus = replace(corpus, states=corpus.goal_states)
    task_head, task_stats = fit(task_corpus, train_initial, initial_turns(corpus, validation_rows), config)
    train_by_family = defaultdict(list)
    for index in train_initial:
        row = corpus.rows[index]
        train_by_family[row["family"]].append(row["outcome"])
    global_rate = sum(corpus.rows[i]["outcome"] for i in train_initial) / len(train_initial)
    family_rates = {key: float(np.mean(outcomes)) for key, outcomes in train_by_family.items()}
    test = corpus.indices("test", args.train_policy)
    predictions = predict_rows(corpus, test, head, task_head, global_rate, family_rates)
    results = {
        "training_policy": args.train_policy,
        "prototype": {"model": MODEL_ID, "revision": MODEL_REVISION, **asdict(context)},
        "hardware": torch.cuda.get_device_name(),
        "fit_config": asdict(config),
        "feature_extraction": corpus.features,
        "coverage": corpus.coverage,
        "value_fit": fit_stats,
        "task_only_fit": task_stats,
        "evaluation": report(predictions, args.output / "calibration.png"),
    }
    if args.update_policy:
        update_pool = corpus.indices("train", args.update_policy)
        update_tasks = sorted(
            {corpus.rows[i]["task_id"] for i in update_pool}, key=lambda task: (stable_hash("update", task), task)
        )[: args.update_tasks]
        update_task_ids = set(update_tasks)
        update = [i for i in update_pool if corpus.rows[i]["task_id"] in update_task_ids]
        new_validation = corpus.indices("validation", args.update_policy)
        new_test = corpus.indices("test", args.update_policy)
        updated_head, update_stats = fit(corpus, update, new_validation, replace(config, epochs=8), initial_head=head)
        before = predict_rows(corpus, new_test, head, task_head, global_rate, family_rates)
        after = predict_rows(corpus, new_test, updated_head, task_head, global_rate, family_rates)
        update_tokens = {corpus.rows[i]["trajectory_id"]: corpus.rows[i]["input_tokens"] for i in update}
        results["policy_drift"] = {
            "target_policy": args.update_policy,
            "update_tasks": len(update_tasks),
            "update_fit": update_stats,
            "before": report(before, args.output / "later-policy-before.png"),
            "after": report(after, args.output / "later-policy-after.png"),
            "update_input_tokens": sum(update_tokens.values()),
        }
        save_checkpoint(args.output / "value-updated.pt", updated_head, context)
        predictions.extend(before)
    save_checkpoint(args.output / "value.pt", head, context)
    save_checkpoint(args.output / "task-only.pt", task_head, context)
    with (args.output / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results["evaluation"]["overall"], indent=2), flush=True)


def load_checkpoint(path: Path, model_dir: str) -> tuple[TurnEncoder, ValueHead, Context]:
    device = torch.device("cuda")
    checkpoint = torch.load(path, weights_only=True)
    if (checkpoint["model"], checkpoint["revision"], checkpoint["encoding_version"]) != (
        MODEL_ID,
        MODEL_REVISION,
        ENCODING_VERSION,
    ):
        raise ValueError("checkpoint uses a different backbone or encoding")
    encoder = TurnEncoder(model_dir, device)
    head = ValueHead(encoder.model.config.hidden_size).to(device).eval()
    head.load_state_dict(checkpoint["state_dict"])
    return (
        encoder,
        head,
        Context(checkpoint["max_tokens"], ContextPolicy(checkpoint.get("context_policy", ContextPolicy.PREFIX_LIMIT))),
    )


def inference(args: argparse.Namespace) -> None:
    """Load a value checkpoint and score every covered turn in the input."""
    encoder, head, context = load_checkpoint(args.checkpoint, args.model_dir)
    with torch.inference_mode(), args.output.open("x") as handle:
        for shard in args.shards:
            for features in feature_rows(encoder, shard, context, 1):
                episode = features.episode
                probabilities = (
                    logits(head, features.states.to(device=encoder.device, dtype=torch.float32)).sigmoid().tolist()
                )
                row = {
                    "trajectory_id": episode.trajectory_id,
                    "probabilities": [
                        {
                            "predict_before": index,
                            "input_tokens": pos + 1,
                            "unpruned_input_tokens": original + 1,
                            "probability": probability,
                        }
                        for index, pos, original, probability in zip(
                            episode.predict_before[: len(features.positions)],
                            features.positions,
                            features.source_positions,
                            probabilities,
                            strict=True,
                        )
                    ],
                }
                handle.write(json.dumps(row) + "\n")


@torch.inference_mode()
def probe(args: argparse.Namespace) -> None:
    """Measure fresh-prefix latency and sensitivity to a changed criterion."""
    encoder, head, context = load_checkpoint(args.checkpoint, args.model_dir)
    tasks = set()
    measurements = []
    alternative = "Submit a valid patch that leaves the repository files unchanged."
    for episode in read_shards(args.shards):
        if episode.split != "test" or episode.task_id in tasks:
            continue
        _, task_text = episode.goal.split("\nRepository:", 1)
        changed = replace(episode, goal=f"{alternative}\nRepository:{task_text}")
        comparisons = [
            [(item, pos) for item, _ in decision_windows(trace, encoder.tokenizer, context) for pos in item.positions]
            for trace in (episode, changed)
        ]
        common = min(map(len, comparisons))
        if common < 2:
            continue
        tasks.add(episode.task_id)
        for turn in (0, common - 1):
            pair = []
            for decisions in comparisons:
                item, pos = decisions[turn]
                single = replace(item, ids=item.ids[: pos + 1], positions=[pos])
                if not measurements and not pair:
                    encoder.hidden([single])
                started = time.perf_counter()
                [(_, state)] = encoder.hidden([single])
                p = logits(head, state.to(device=encoder.device, dtype=torch.float32)).sigmoid().item()
                pair.append((p, time.perf_counter() - started, pos + 1))
            measurements.append(
                {
                    "task_id": episode.task_id,
                    "turn": turn,
                    "original_input_tokens": pair[0][2],
                    "original_seconds": pair[0][1],
                    "original_probability": pair[0][0],
                    "alternative_probability": pair[1][0],
                }
            )
        if len(tasks) == args.tasks:
            break
    args.output.write_text(json.dumps(measurements, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command, function in (("fit", train), ("predict", inference), ("probe", probe)):
        subcommand = subcommands.add_parser(command)
        subcommand.add_argument("--model-dir", default=MODEL_ID)
        subcommand.add_argument("--output", type=Path, required=True)
        subcommand.add_argument("shards", type=Path, nargs="+")
        subcommand.set_defaults(function=function)
    for command in ("predict", "probe"):
        subcommands.choices[command].add_argument("--checkpoint", type=Path, required=True)
    subcommands.choices["probe"].add_argument("--tasks", type=int, default=12)
    training = subcommands.choices["fit"]
    training.add_argument("--cache-dir", type=Path, required=True)
    training.add_argument("--train-policy", required=True)
    training.add_argument("--update-policy")
    training.add_argument("--update-tasks", type=int, default=24)
    training.add_argument("--max-tokens", type=int, required=True)
    training.add_argument("--context-policy", choices=list(ContextPolicy), default=ContextPolicy.PREFIX_LIMIT)
    training.add_argument("--encode-batch-size", type=int, default=2)
    training.add_argument("--head-batch-size", type=int, default=256)
    training.add_argument("--lr", type=float, default=0.002)
    training.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
