# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Propose a weight-only T3 input packet from immutable T2 native cache bytes."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

from experiments.post_training.identity_t2.prepare import sha, summarize
from experiments.post_training.identity_t2.relocate import encoded, relocate_paths
from experiments.post_training.identity_t2.sampling_preview import preview

T2_PACKET_SHA = "ae730c5772a3cfc3e2783ebc726ebcafa4f4ff7a3b3322e7b644ab4f7c01ae1c"
WEIGHTS = {"identity": 264.0, "preservation": 547.0, "identity_v2": 504.0, "tools": 312.0, "third_person": 324.0}
ROW_WEIGHTS = {"identity": 1.2, "preservation": 1.0, "identity_v2": 4.5, "tools": 2.0, "third_person": 6.0}


def propose(source: Path, parent_packet: Path, sampler: Path, out: Path, packet: Path) -> dict:
    """Copy only hash-bound small inputs; replay native sampling before requesting review."""
    code_pin = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert sha(parent_packet.read_bytes()) == T2_PACKET_SHA
    parent = json.loads(parent_packet.read_bytes())
    for filename, digest in parent["source_artifacts"].items():
        assert sha(Path(filename).read_bytes()) == digest
    report = json.loads((sampler / "sampling-preview-report.json").read_bytes())
    assert report["component_weights"] == WEIGHTS and report["first128"]["identity_token_share"] <= 0.08
    original_config = json.loads((source / "data-config.json").read_bytes())
    config = json.loads((sampler / "data-config-draft.json").read_bytes())
    assert {**config, "train_weights": original_config["train_weights"]} == original_config
    assert config["train_weights"] == WEIGHTS
    assert not out.exists() and not packet.exists()
    prefixes = {str(source): str(out)}
    payloads = {}
    for name, info in parent["new_native_cache_files"].items():
        raw = (source / name).read_bytes()
        assert sha(raw) == info["sha256"]
        payloads[name] = raw
    for name in (
        "identity_v2.jsonl",
        "tools.jsonl",
        "third_person.jsonl",
        "native-row-fingerprints.json",
        "native-validity-all-new.json",
    ):
        payloads[name] = (source / name).read_bytes()
    payloads["cache-catalog.json"] = encoded(
        relocate_paths(json.loads((source / "cache-catalog.json").read_bytes()), exact={}, prefixes=prefixes)
    )
    payloads["data-config.json"] = encoded(relocate_paths(config, exact={}, prefixes=prefixes))
    occurrences_raw = (sampler / "occurrences-128.jsonl").read_bytes()
    occurrences = [json.loads(line) for line in occurrences_raw.splitlines()]
    assert len(occurrences) == 128 and summarize(occurrences) == report["first128"]
    payloads["occurrences-128.jsonl"] = occurrences_raw
    share = json.loads((source / "share-preview.json").read_bytes())
    share.update(
        {
            "sampling_weights": WEIGHTS,
            "per_row_weights": ROW_WEIGHTS,
            "first128": report["first128"],
            "per_update": [summarize(occurrences[i : i + 8]) for i in range(0, 128, 8)],
            "cache_catalog_sha256": sha(payloads["cache-catalog.json"]),
            "state": "PROPOSED_T3_WEIGHTS_ONLY_NOT_REVIEWED",
        }
    )
    for field in ("per_identity_row_sampling_weight", "control_and_preservation_row_sampling_weight"):
        del share[field]
    share["mean_update_identity_token_fraction"] = (
        sum(update["identity_token_share"] for update in share["per_update"]) / 16
    )
    payloads["share-preview.json"] = encoded(share)
    out.mkdir(parents=True)
    for name, raw in payloads.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    # Keep the native-input builder pin distinct from the new sampling/preparation pin.
    proposed = {
        "scope": "PROPOSED T3 freeze correctness review; NOT frozen/mirrored/training-approved",
        "prepared_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "code_pin": code_pin,
        "state": "PROPOSED_T3_NOT_REVIEWED_OR_REGIONAL",
        "source_code": {
            name: sha(Path(name).read_bytes())
            for name in (
                "experiments/post_training/identity_t2/sampling_preview.py",
                "experiments/post_training/identity_t2/freeze_sampling.py",
            )
        },
        "old_rows": parent["old_rows"],
        "new_counts": parent["new_counts"],
        "combined_rows": parent["combined_rows"],
        "dispositions": parent["dispositions"],
        "native_input_support": parent["native_input_support"],
        "new_native_cache_files": parent["new_native_cache_files"],
        "current_orders_sha256": sha(
            Path("/home/karan/Dropbox/marin-notes/snowball-identity/t310/current-orders.md").read_bytes()
        ),
        "regional_freeze": {
            "proposed_root": "s3://marin-rno2a/marin/experiments/snowball-identity/t312/cd16-t3/freeze-r1",
            "not_written": True,
        },
        "freeze_approved": False,
        "training_approved": False,
        "native_input_lineage_code_pin": parent["code_pin"],
        "parent_logical_packet_sha256": T2_PACKET_SHA,
        "parent_logical_packet_path": str(parent_packet),
        "sampling_weights": WEIGHTS,
        "per_row_weights": ROW_WEIGHTS,
        "first128": report["first128"],
        "first128_component_family_sightings": report["first128_component_family_sightings"],
        "first128_component_token_shares": report["first128_component_supervised_token_shares"],
        "new_identity_sightings": report["new_identity_sightings"],
        "new_identity_unique_ids": report["new_identity_unique_ids"],
        "source_artifacts": {
            **parent["source_artifacts"],
            **{str(out / name): sha(raw) for name, raw in payloads.items()},
            **{str(p): sha(p.read_bytes()) for p in sampler.iterdir() if p.is_file()},
        },
        "source_changes": "Only sampling weights and explicit local storage coordinates",
        "GPU_or_job_actions": False,
    }
    packet.write_bytes(encoded(proposed))
    # Real native replay from the newly proposed paths, not a projection of expected counts.
    replay = preview(out, out.parent / (out.name + "-native-replay"), 4.5, packet)
    assert replay["first128"] == report["first128"]
    assert (out.parent / (out.name + "-native-replay") / "occurrences-128.jsonl").read_bytes() == occurrences_raw
    for name, raw in payloads.items():
        assert (out / name).read_bytes() == raw
    print(
        json.dumps(
            {
                "packet": str(packet),
                "sha256": sha(packet.read_bytes()),
                "code_pin": code_pin,
                "first128": report["first128"],
                "native_replay_exact": True,
            },
            indent=2,
        )
    )
    return proposed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent-packet", type=Path, required=True)
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    propose(args.source, args.parent_packet, args.sampler, args.out, args.packet)


if __name__ == "__main__":
    main()
