# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble complete eligible seed/arm results for the predeclared inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def assemble(manifest_path: Path, protocol_path: Path, seeds: list[int], output: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    allowed = protocol["initial_seeds"] + protocol["extension"]["maximum_additional_matched_seeds"]
    if seeds != allowed[: len(seeds)] or not 3 <= len(seeds) <= 6:
        raise ValueError("Only the predeclared initial seeds and bounded extension can be assembled")
    arms = ["tis", "sc128", "sc32"]
    tasks, families, fingerprints = None, None, None
    success, scores, seconds = [], [], []
    sources = []
    endpoint = protocol["training"]["planned_updates"]
    for seed in seeds:
        seed_success, seed_scores, seed_seconds = [], [], []
        for arm in arms:
            matches = [row for row in manifest["runs"] if row["seed"] == seed and row["arm"] == arm and row["eligible"]]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one eligible run for {seed}/{arm}")
            run = matches[0]
            root = manifest_path.parent / run["collected"]
            summary_path = root / "summary.json"
            grade_path = root / f"grades-{endpoint:04d}.jsonl"
            summary = json.loads(summary_path.read_text())
            grades = [json.loads(line) for line in grade_path.read_text().splitlines()]
            quality_rows = protocol["evaluation"].get("quality_rows", protocol["split"]["heldout_rows"])
            if len(grades) != quality_rows or len(summary["cycles"]) != endpoint:
                raise ValueError("Incomplete endpoint or policy-update window")
            if summary.get("eligibility_sha256") != protocol["evaluation"].get("eligibility_sha256"):
                raise ValueError("Run used a different quality eligibility rule")
            identities = [row["task"] for row in grades]
            domains = [row["family"] for row in grades]
            public = [row["public_fingerprint"] for row in grades]
            if tasks is None:
                tasks, families, fingerprints = identities, domains, public
            if (identities, domains, public) != (tasks, families, fingerprints):
                raise ValueError("The paired held-out panel changed between runs")
            if any(row["full_success"] is None or row["normalized_verifier_score"] is None for row in grades):
                raise ValueError("Missing grades cannot become failures")
            seed_success.append([row["full_success"] for row in grades])
            seed_scores.append([row["normalized_verifier_score"] for row in grades])
            seed_seconds.append(summary["training_cycle_seconds"])
            sources.append(
                {
                    **run,
                    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    "grades_sha256": hashlib.sha256(grade_path.read_bytes()).hexdigest(),
                }
            )
        success.append(seed_success)
        scores.append(seed_scores)
        seconds.append(seed_seconds)
    np.savez_compressed(
        output,
        seeds=np.array(seeds),
        arms=np.array(arms),
        tasks=np.array(tasks),
        families=np.array(families),
        success=np.array(success),
        scores=np.array(scores),
        training_seconds=np.array(seconds),
        updates=np.full((len(seeds), len(arms)), endpoint),
    )
    output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "npz_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1701, 2903, 4517])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = parser.parse_args()
    assemble(args.manifest, args.protocol, args.seeds, args.output)
