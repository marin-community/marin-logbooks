# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind B-base to the reviewed native r4 pool, without generated reasoning."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.post_training.identity_path_b.exposure import GLOBAL_BATCH, SEQUENCE_LENGTH, pack_arm, target_counts

IDENTITY_SHA256 = "0d0fd663593786ea356e843eb0a64d684778174c9585d25110b142b8c5f84930"
BASE_BINDING = {
    "experiment": "B-base",
    "identity_pool_sha256": IDENTITY_SHA256,
    "identity_original_sources": 40978,
    "identity_rendered_rows": 81956,
    "identity_unique_final_text_tokens": 843552,
    "identity_reasoning": "native_r4_explicit_empty_on_span",
}
ARMS = {"10": (0.10, 500000), "25": (0.25, 1500000)}


def load_base_identity(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != IDENTITY_SHA256:
        raise ValueError("B-base requires the exact reviewed native r4 identity pool")
    rows = [json.loads(line) for line in raw.splitlines()]
    pairs = defaultdict(dict)
    for row in rows:
        mode = row["enable_thinking"]
        if mode in pairs[row["source_id"]]:
            raise ValueError("Repeated B-base source/mode")
        pairs[row["source_id"]][mode] = row
    if len(rows) != BASE_BINDING["identity_rendered_rows"] or len(pairs) != BASE_BINDING["identity_original_sources"]:
        raise ValueError("B-base source/row census changed")
    lexical = 0
    for pair in pairs.values():
        if set(pair) != {False, True}:
            raise ValueError("B-base source lacks its native off/on pair")
        off, on = pair[False], pair[True]
        targets = [row["input_ids"][row["target_start"] :] for row in (off, on)]
        if (
            on["messages"][:-1] != off["messages"][:-1]
            or on["messages"][-1]["content"] != "<|start_think|><|end_think|>" + off["messages"][-1]["content"]
            or targets[1] != [128002, 128003, *targets[0]]
        ):
            raise ValueError("B-base must preserve canonical finals and empty ON reasoning")
        if target_counts(on).reasoning_text != 0:
            raise ValueError("B-base cannot contain generated reasoning")
        lexical += target_counts(off).final_text
    if lexical != BASE_BINDING["identity_unique_final_text_tokens"]:
        raise ValueError("B-base unique final-text census changed")
    return rows


def check_base_exposure(exposure: dict, arm: str) -> None:
    """Refuse a different identity variant or dose in the B-base worker."""
    if any(exposure.get(key) != value for key, value in BASE_BINDING.items()):
        raise ValueError("Worker exposure is not bound to the reviewed B-base identity pool")
    share, minimum = ARMS[arm]
    if exposure["identity_share_target"] != share or exposure["identity_minimum_excludes_structural_tokens"] != minimum:
        raise ValueError("B-base arm share/minimum changed")
    if exposure["batch_size"] != GLOBAL_BATCH or exposure["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("B-base packed batch dimensions changed")
    if exposure["final_text_tokens"]["identity"] < minimum or exposure["reasoning_text_tokens"].get("identity", 0):
        raise ValueError("B-base requires the final-text minimum and zero identity reasoning text")
    if exposure["maximum_identity_source_occurrences_across_modes"] > 3:
        raise ValueError("B-base source conversation exceeds three total occurrences")


def build(identity: Path, replay: Path, arm: str, output: Path) -> dict:
    identity_rows = load_base_identity(identity)
    replay_bytes = replay.read_bytes()
    replay_rows = [json.loads(line) for line in replay_bytes.splitlines()]
    share, minimum = ARMS[arm]
    report = pack_arm(identity_rows, replay_rows, identity_share=share, final_identity_minimum=minimum, output=output)
    report.update(BASE_BINDING)
    report["replay_pool_sha256"] = hashlib.sha256(replay_bytes).hexdigest()
    report["replay_family_rows"] = dict(Counter(row["family"] for row in replay_rows))
    check_base_exposure(report, arm)
    (output / "exposure.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.identity, args.replay, args.arm, args.output)
    print(json.dumps({key: report[key] for key in (*BASE_BINDING, "updates", "replay_pool_sha256")}, indent=2))


if __name__ == "__main__":
    main()
