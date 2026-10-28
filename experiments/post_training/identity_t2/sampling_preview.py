# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only T3 sampling preview over unchanged T2 caches; no staging or jobs."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from itertools import batched
from pathlib import Path

import draccus
import haliax as hax
import jax
import numpy as np
from levanter.data.text.datasets import LmDataConfig
from levanter.schedule import BatchSchedule

from experiments.post_training.identity_t2.prepare import (
    BATCH_SIZE,
    SEED,
    SEQ_LEN,
    array_sha,
    check_validity,
    sha,
    summarize,
)


def preview(source: Path, out: Path, new_identity_weight: float, binding_packet: Path) -> dict:
    if jax.default_backend() != "cpu":
        raise ValueError("Only CPU sampling preparation is authorized")
    source_files = {
        name: (source / name).read_bytes()
        for name in (
            "data-config.json",
            "cache-catalog.json",
            "native-row-fingerprints.json",
            "share-preview.json",
        )
    }
    authority = json.loads(binding_packet.read_bytes())
    for name, raw in source_files.items():
        assert sha(raw) == authority["source_artifacts"][str(source / name)]
    original = draccus.decode(LmDataConfig, json.loads(source_files["data-config.json"]))
    old_root = Path(original.components["identity"].cache_dir).parents[1]
    old_manifest = json.loads((old_root / "manifest.json").read_bytes())
    assert sha((old_root / "manifest.json").read_bytes()) == authority["old_rows"]["manifest_sha256"]
    cache_checks = {old_root / name: info["sha256"] for name, info in old_manifest["cache_files"].items()}
    cache_checks.update({source / name: info["sha256"] for name, info in authority["new_native_cache_files"].items()})
    assert len(cache_checks) == 45
    for path, digest in cache_checks.items():
        assert sha(path.read_bytes()) == digest
    metadata = json.loads(source_files["native-row-fingerprints.json"])
    assert len(metadata) == 1089
    assert Counter(row["component"] for row in metadata.values()) == {
        "identity": 220,
        "preservation": 547,
        "identity_v2": 112,
        "tools": 156,
        "third_person": 54,
    }
    assert original.auto_build_caches is False and original.permutation_type == "feistel"
    assert original.mixture_block_size == 2048
    assert new_identity_weight == 4.5
    weights = {**original.train_weights, "identity_v2": 112 * new_identity_weight, "tools": 468.0, "third_person": 324.0}
    assert weights["identity"] == 264.0 and weights["preservation"] == 547.0
    data = dataclasses.replace(original, train_weights=weights)
    assert dataclasses.replace(data, train_weights=original.train_weights) == original
    lookup = {(row["tokens_sha256"], row["target_mask_sha256"]): identifier for identifier, row in metadata.items()}
    assert len(lookup) == 1089
    native = data.train_set(
        hax.Axis("position", SEQ_LEN), BatchSchedule(BATCH_SIZE), key=jax.random.PRNGKey(SEED)
    ).as_sync_dataset()
    occurrences, validity = [], []
    for positions in batched(range(128), BATCH_SIZE):
        for position, example in zip(positions, native.get_batch(list(positions)), strict=True):
            tokens, mask = np.asarray(example.tokens.array), np.asarray(example.loss_weight.array)
            assert tokens.shape == mask.shape == (SEQ_LEN,) and np.all((mask == 0) | (mask == 1))
            identifier = lookup[(array_sha(tokens), array_sha(mask))]
            meta = metadata[identifier]
            assert int(mask.sum()) == meta["supervised_tokens"]
            if "valid_input_length" in meta:
                query, keys = example.attn_mask.segment_ids
                validity.append(check_validity(query.array, keys.array, meta))
            occurrences.append({"position": position, "update": position // BATCH_SIZE, **meta})
    families = defaultdict(Counter)
    component_tokens = Counter()
    for row in occurrences:
        families[row["component"]][row["family"]] += 1
        component_tokens[row["component"]] += row["supervised_tokens"]
    families["identity_v2"].setdefault("history", 0)
    totals = summarize(occurrences)
    new_identity = [row for row in occurrences if row["component"] == "identity_v2"]
    report = {
        "scope": "CPU-only draft sampling preview; unchanged1089 rows/caches/recipe; NOT frozen or reviewed",
        "source_files_sha256": {name: sha(raw) for name, raw in source_files.items()},
        "source_root": str(source),
        "rows": 1089,
        "seed": SEED,
        "permutation": "feistel",
        "block": 2048,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQ_LEN,
        "component_order": list(data.components),
        "component_weights": weights,
        "per_row_sampling_weights": {
            "old_identity": 1.2,
            "new_identity_v2": new_identity_weight,
            "tools": 3.0,
            "third_person_controls": 6.0,
            "preservation": 1.0,
        },
        "first128": totals,
        "first128_component_family_sightings": {key: dict(value) for key, value in families.items()},
        "first128_component_supervised_tokens": dict(component_tokens),
        "first128_component_supervised_token_shares": {
            key: value / totals["supervised_tokens"] for key, value in component_tokens.items()
        },
        "new_identity_sightings": len(new_identity),
        "new_identity_unique_ids": len({row["id"] for row in new_identity}),
        "new_identity_sighting_target": 32,
        "identity_share_limit": 0.08,
        "all_45_T2_cache_files_hash_verified_before_and_after": True,
        "identity_share_le_8pct": totals["identity_token_share"] <= 0.08,
        "loss_masks_binary_and_unchanged": True,
        "new_selected_actual_segment_checks": len(validity),
        "freeze_approved": False,
        "freeze_review_requested": False,
        "regional_writes": False,
        "GPU_or_submit_actions": False,
    }
    for name, raw in source_files.items():
        assert (source / name).read_bytes() == raw
    for path, digest in cache_checks.items():
        assert sha(path.read_bytes()) == digest
    assert not out.exists()
    out.mkdir(parents=True)
    (out / "data-config-draft.json").write_text(json.dumps(draccus.encode(data), indent=2) + "\n")
    (out / "occurrences-128.jsonl").write_text("".join(json.dumps(row) + "\n" for row in occurrences))
    (out / "selected-new-validity.json").write_text(json.dumps(validity, indent=2) + "\n")
    (out / "sampling-preview-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--new-identity-weight", required=True, type=float)
    parser.add_argument("--binding-packet", required=True, type=Path)
    args = parser.parse_args()
    preview(args.source, args.out, args.new_identity_weight, args.binding_packet)


if __name__ == "__main__":
    main()
