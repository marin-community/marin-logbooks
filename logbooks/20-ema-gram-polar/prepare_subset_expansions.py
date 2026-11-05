# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare frozen subset LR expansions from complete groups; cannot submit jobs."""

import argparse
import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import draccus
from marin.execution.artifact import is_mutable_version, validate_version
from marin.execution.lazy import materialized_config
from marin.training.training import apply_output_path
from submit_subset_confirmations import validate_completions

from experiments.ema_gram.plan import Cell, Result, evaluated_groups, expand
from experiments.ema_gram.subset_campaign import BASE, CELLS, VERSION, prepared, source_hashes
from experiments.ema_gram.train import build_training_step


def next_cells(records):
    """Replay complete groups from the original grid, including prior expansions."""
    for identity in ("job_id", "run_id", "run_url"):
        if len({r[identity] for r in records["results"]}) != len(records["results"]):
            raise ValueError("Completion identity reused across cells: " + identity)
    results = [
        Result(
            Cell(**r["cell"]).name,
            "completed",
            r["final_c4_loss"],
            r["stored_native_counter"],
            r["expected_steps"],
            r["run_url"],
        )
        for r in records["results"]
    ]
    validate_completions(SimpleNamespace(results=results), records)
    by_name = {r.name: r for r in results}
    consumed, proposed, groups = set(), [], []
    for alpha, ridge in sorted({(c.alpha, c.ridge) for c in CELLS}):
        group = [c for c in CELLS if (c.alpha, c.ridge) == (alpha, ridge)]
        rounds = 0
        while True:
            names = {c.name for c in group}
            consumed.update(names & by_name.keys())
            missing = [c for c in group if c.name not in by_name]
            if missing:
                # Initial pilots already own permanent launch identities.
                if rounds:
                    proposed.extend(missing)
                groups.append(
                    dict(
                        alpha=alpha,
                        ridge=ridge,
                        rounds=rounds,
                        state="awaiting_expansion" if rounds else "awaiting_initial_pilots",
                        missing=[c.name for c in missing],
                    )
                )
                break
            group_results = [by_name[c.name] for c in group]
            updated = expand(group, group_results)
            if updated == group:
                winner = evaluated_groups(group, group_results)[0][1]
                groups.append(
                    dict(alpha=alpha, ridge=ridge, rounds=rounds, state="interior_winner", winner=asdict(winner))
                )
                break
            group = updated
            rounds += 1
    if consumed != set(by_name):
        raise ValueError("Results contain unplanned or premature expansion cells")
    return proposed, groups


def accepted_bindings(records, accepted_expansions):
    initial = json.loads((BASE / "subset-training-r02-launched.json").read_text())["ledger"]["cells"].values()
    entries = list(initial) + list(accepted_expansions)
    by_name = {Cell(**entry["cell"]).name: entry for entry in entries}
    if len(by_name) != len(entries) or len({e["job_id"] for e in entries}) != len(entries):
        raise ValueError("Repeated accepted cell or job binding")
    for entry in entries:
        if not entry["launch_started"] or not re.fullmatch("[0-9a-f]{40}", entry["source_commit"]):
            raise ValueError("Missing permanent guard or exact source binding")
    for row in records["results"]:
        name = Cell(**row["cell"]).name
        if name not in by_name:
            raise ValueError("Completed cell has no accepted launch binding")
        entry = by_name[name]
        if row["source"] != entry["source_commit"] or row["job_id"] != entry["job_id"]:
            raise ValueError("Completed source/job differs from accepted launch")
        if not entry["launched"] or entry["submit_exit_code"] != 0:
            raise ValueError("Ambiguous accepted launch requires reconciliation")
        if not row["run_id"].startswith(name + "-"):
            raise ValueError("Native run identity does not belong to completed cell")
    return by_name


def prepare(records, version, accepted_expansions=()):
    validate_version(version)
    if is_mutable_version(version) or version == VERSION:
        raise ValueError("Use a fresh distinct immutable expansion artifact version")
    bindings = accepted_bindings(records, accepted_expansions)
    validation = json.loads((BASE / "subset-training-validation.json").read_text())
    hashes = source_hashes()
    if not validation["passed"] or validation["source_files"] != hashes:
        raise ValueError("Training numerical sources differ from qualification")
    for name in ("subset-inputs.json", "subset-dataset.json", "subset-cache-manifest.json"):
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != validation["artifacts"][name]:
            raise ValueError("Frozen subset artifact changed: " + name)
    candidates, groups = next_cells(records)
    rows, guarded = [], []
    for cell in candidates:
        if cell.name in bindings:
            guarded.append(cell.name)
            continue
        variant = "v4-8" if cell.alpha == 0 else "v4-32"
        template = prepared(0, variant)
        step = build_training_step(
            cell,
            version,
            tpu_type=variant,
            host_memory="256g",
            eigh_layout="matrix",
            eigh_implementation="default",
            initialize_from=None,
            profile_start_step=100,
            profile_steps=1,
        )
        pod = materialized_config(step, "gs://marin-us-central2")
        config = replace(apply_output_path(pod.train_config, pod.output_path), data=template.train_config.data)
        if config.trainer.num_train_steps != 7630 or config.trainer.train_batch_size != 8:
            raise ValueError("Expansion must retain the complete pilot schedule")
        if config.trainer.initialize_from is not None or config.initialize_from_checkpoint_path is not None:
            raise ValueError("Expansion must start from scratch")
        if config.trainer.seed != 0 or config.data_seed != 0 or config.trainer.tracker.mode != "online":
            raise ValueError("Expansion requires seed0 and ONLINE tracking")
        if config.optimizer.eigh_layout != "matrix" or config.optimizer.eigh_implementation != "default":
            raise ValueError("Expansion optimizer differs from qualified implementation")
        encoded = draccus.encode(replace(pod, train_config=config))
        rows.append(
            dict(
                cell=asdict(cell),
                variant=variant,
                config=encoded,
                config_sha256=hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest(),
            )
        )
    return dict(
        artifact_version=version,
        groups=groups,
        cells=rows,
        guarded_expansions=guarded,
        source_files=hashes,
        accepted_expansions_sha256=hashlib.sha256(json.dumps(accepted_expansions, sort_keys=True).encode()).hexdigest(),
        completion_records_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        submitted=False,
        launch_ready=False,
        remaining_gates=[
            "Published completion/source/job bindings for every predecessor",
            "Fresh execution worktree, campaign identities and durable launch guards",
            "Qualified worker with fresh ONLINE logging before kernels",
            "Full cache catalog check and one exclusive observer",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--accepted-expansions", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.completions.read_text())
    accepted = json.loads(args.accepted_expansions.read_text()) if args.accepted_expansions else []
    bundle = prepare(records, args.version, accepted)
    with args.output.open("x") as stream:
        stream.write(json.dumps(bundle, indent=2) + "\n")


if __name__ == "__main__":
    main()
