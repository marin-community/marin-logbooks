# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exclusive in-region staging/readback of small frozen data only; never submit jobs."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from itertools import batched
from pathlib import Path

import draccus
import haliax as hax
import jax
import numpy as np
from levanter.data.text.datasets import LmDataConfig
from levanter.schedule import BatchSchedule
from rigging.filesystem.buckets import filesystem_for
from rigging.fsutil.listing import list_entries

from experiments.post_training.identity_t2.prepare import BATCH_SIZE, SEED, SEQ_LEN, array_sha, check_validity, summarize

R8_URI = "s3://marin-rno2a/marin/experiments/snowball-identity/t310/lr0-r8/lr0-result.json"
R8_SHA = "eab9e42ed3d16284b7f2bf7c979c6dfc5b448bcaa78e99c3a89f87537e350ec4"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    # Component insertion order assigns shuffle keys; never recursively sort.
    return (json.dumps(value, indent=2) + "\n").encode()


def relocate_paths(value, *, exact: dict[str, str], prefixes: dict[str, str]):
    """Change only explicit storage coordinates, preserving component/shuffle order."""
    if isinstance(value, str):
        if value in exact:
            return exact[value]
        for source, destination in prefixes.items():
            if value.startswith(source + "/"):
                return destination + value[len(source) :]
    if isinstance(value, dict):
        return {key: relocate_paths(item, exact=exact, prefixes=prefixes) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_paths(item, exact=exact, prefixes=prefixes) for item in value]
    return value


def fresh_root(root: str) -> dict:
    fs, key = filesystem_for(root)
    if fs.exists(key) or list_entries(root, recursive=True):
        raise FileExistsError("Refusing existing or partial regional data root")
    ancestor = root.rsplit("/", 1)[0]
    missing = []
    while True:
        try:
            entries = list_entries(ancestor)
            return {
                "root": root,
                "VOID": True,
                "missing_parents": missing,
                "listed_ancestor": ancestor,
                "ancestor_entries": len(entries),
            }
        except FileNotFoundError:
            afs, akey = filesystem_for(ancestor)
            assert not afs.exists(akey)
            missing.append(ancestor)
            ancestor = ancestor.rsplit("/", 1)[0]
            assert ancestor.startswith("s3://marin-rno2a/")


def run(
    old: Path, preview: Path, root: str, out: Path, logical_packet: Path, approved_logical_packet_sha256: str
) -> dict:
    if jax.default_backend() != "cpu" or not root.startswith("s3://marin-rno2a/"):
        raise ValueError("Only CPU data staging to RNO is authorized")
    assert not out.exists()
    assert sha(logical_packet.read_bytes()) == approved_logical_packet_sha256
    authority = json.loads(logical_packet.read_bytes())
    expected = json.loads((preview / "share-preview.json").read_bytes())
    assert expected["first128"]["identity_token_share"] <= 0.08
    assert expected["sampling_weights"] == authority["sampling_weights"]
    assert expected["per_row_weights"] == authority["per_row_weights"]
    assert expected["sampling_weights"] == {
        group: count * expected["per_row_weights"][group]
        for group, count in (
            ("identity", 220),
            ("preservation", 547),
            ("identity_v2", 112),
            ("tools", 156),
            ("third_person", 54),
        )
    }
    for name in (
        "share-preview.json",
        "data-config.json",
        "cache-catalog.json",
        "native-row-fingerprints.json",
        "occurrences-128.jsonl",
        "native-validity-all-new.json",
    ):
        assert sha((preview / name).read_bytes()) == authority["source_artifacts"][str(preview / name)]
    old_manifest = json.loads((old / "manifest.json").read_bytes())
    assert sha((old / "manifest.json").read_bytes()) == authority["old_rows"]["manifest_sha256"]
    old_config = json.loads((old / "data-config.json").read_bytes())
    old_inputs = {
        name: Path(old_config["components"][name]["source"]["train_urls"][0]) for name in ("identity", "preservation")
    }
    mappings = {str(preview / "cache-catalog.json"): root + "/cache-catalog.json"}
    mappings.update({str(path): root + "/inputs/base-" + name + ".jsonl" for name, path in old_inputs.items()})

    prefixes = {str(preview): root + "/new", str(old): root + "/base"}
    config = relocate_paths(json.loads((preview / "data-config.json").read_bytes()), exact=mappings, prefixes=prefixes)
    catalog = relocate_paths(
        json.loads((preview / "cache-catalog.json").read_bytes()), exact=mappings, prefixes=prefixes
    )
    assert list(config["components"]) == expected["component_order"]
    assert config["train_weights"] == expected["sampling_weights"]
    payloads = {}
    for relative, info in old_manifest["cache_files"].items():
        raw = (old / relative).read_bytes()
        assert sha(raw) == info["sha256"]
        payloads["base/" + relative] = raw
    for relative, info in authority["new_native_cache_files"].items():
        raw = (preview / relative).read_bytes()
        assert sha(raw) == info["sha256"]
        payloads["new/" + relative] = raw
    for name, path in old_inputs.items():
        raw = path.read_bytes()
        assert sha(raw) == old_manifest["source_files"][str(path)]
        payloads["inputs/base-" + name + ".jsonl"] = raw
    candidates_path = next(
        Path(name) for name in authority["source_artifacts"] if name.endswith("proposed-post-disposition.jsonl")
    )
    candidates_raw = candidates_path.read_bytes()
    assert sha(candidates_raw) == authority["source_artifacts"][str(candidates_path)]
    candidates = [json.loads(line) for line in candidates_raw.splitlines()]
    for group in ("identity_v2", "tools", "third_person"):
        raw = (preview / (group + ".jsonl")).read_bytes()
        assert (
            raw == "".join(json.dumps(row) + "\n" for row in candidates if row["candidate"]["group"] == group).encode()
        )
        payloads["new/" + group + ".jsonl"] = raw
    for name in (
        "native-row-fingerprints.json",
        "occurrences-128.jsonl",
        "share-preview.json",
        "native-validity-all-new.json",
    ):
        payloads[name] = (preview / name).read_bytes()
    payloads["original-local-manifest.json"] = (old / "manifest.json").read_bytes()
    payloads["logical-freeze-review-packet.json"] = logical_packet.read_bytes()
    for source, digest in old_manifest["source_files"].items():
        assert sha(Path(source).read_bytes()) == digest
    native = json.loads(payloads["native-row-fingerprints.json"])
    assert len(native) == 1089
    payloads["data-config.json"] = encoded(config)
    payloads["cache-catalog.json"] = encoded(catalog)
    guard = fresh_root(root)
    fs, key = filesystem_for(R8_URI)
    assert sha(fs.cat_file(key)) == R8_SHA
    out.mkdir(parents=True, exist_ok=False)
    files = {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in payloads.items()}
    plan = {
        "state": "pre-upload",
        "root": root,
        "freshness": guard,
        "R8_positive_sha256": R8_SHA,
        "files": files,
        "model_payload_operations": False,
        "GPU_or_job_actions": False,
    }
    (out / "staging-plan.json").write_bytes(encoded(plan))
    for name, raw in payloads.items():
        fs, key = filesystem_for(root + "/" + name)
        assert not fs.exists(key)
        fs.pipe_file(key, raw)
        assert sha(fs.cat_file(key)) == files[name]["sha256"]
    (out / "all-bytes-readback.json").write_bytes(encoded({**plan, "state": "all-staged-bytes-identical"}))
    data = draccus.decode(LmDataConfig, json.loads(payloads["data-config.json"]))
    lookup = {(row["tokens_sha256"], row["target_mask_sha256"]): identifier for identifier, row in native.items()}
    assert len(lookup) == 1089
    expected_order = [json.loads(line) for line in payloads["occurrences-128.jsonl"].splitlines()]
    observed = []
    observed_validity = []
    dataset = data.train_set(
        hax.Axis("position", SEQ_LEN), BatchSchedule(BATCH_SIZE), key=jax.random.PRNGKey(SEED)
    ).as_sync_dataset()
    for positions in batched(range(128), BATCH_SIZE):
        for position, example in zip(positions, dataset.get_batch(list(positions)), strict=True):
            tokens, weights = np.asarray(example.tokens.array), np.asarray(example.loss_weight.array)
            identifier = lookup[(array_sha(tokens), array_sha(weights))]
            meta = native[identifier]
            row = {"position": position, "update": position // BATCH_SIZE, **meta}
            assert row == expected_order[position]
            query, keys = example.attn_mask.segment_ids
            length = meta.get("valid_input_length", int((np.asarray(query.array) >= 0).sum()))
            checked = check_validity(query.array, keys.array, {**meta, "valid_input_length": length})
            observed_validity.append(
                {
                    **checked,
                    "query_segments_sha256": array_sha(query.array),
                    "key_segments_sha256": array_sha(keys.array),
                }
            )
            observed.append(row)
    assert summarize(observed) == expected["first128"]
    all_validity = []
    datasets = data.train_sets(
        hax.Axis("position", SEQ_LEN), initial_batch_size=BATCH_SIZE, key=jax.random.PRNGKey(SEED)
    )
    for group, count in (
        ("identity", 220),
        ("preservation", 547),
        ("identity_v2", 112),
        ("tools", 156),
        ("third_person", 54),
    ):
        source = datasets[group].as_sync_dataset()
        for indices in batched(range(count), BATCH_SIZE):
            for example in source.get_batch(list(indices)):
                identifier = lookup[(array_sha(example.tokens), array_sha(example.loss_weight))]
                meta = native[identifier]
                assert meta["component"] == group
                query, key_segments = example.attn_mask.segment_ids
                # Baseline lengths come from actual native packing segments, never PAD/EOS or loss weights.
                # New rows additionally bind to independent captured prefix+target lengths.
                length = meta.get("valid_input_length", int((np.asarray(query) >= 0).sum()))
                checked = check_validity(query, key_segments, {**meta, "valid_input_length": length})
                assert int(np.asarray(example.loss_weight).sum()) == meta["supervised_tokens"]
                all_validity.append(
                    {
                        **checked,
                        "query_segments_sha256": array_sha(query),
                        "key_segments_sha256": array_sha(key_segments),
                    }
                )
    assert len(all_validity) == len({row["id"] for row in all_validity}) == 1089
    by_id = {row["id"]: row for row in all_validity}
    assert observed_validity == [by_id[row["id"]] for row in observed]
    expected_validity = json.loads(payloads["native-validity-all-new.json"])
    new_validity = [
        {key: value for key, value in row.items() if not key.endswith("sha256")}
        for row in all_validity
        if "valid_input_length" in native[row["id"]]
    ]
    assert sorted(new_validity, key=lambda row: row["id"]) == sorted(expected_validity, key=lambda row: row["id"])
    raw_validity = encoded(all_validity)
    fs, key = filesystem_for(root + "/all-native-validity.json")
    assert not fs.exists(key)
    fs.pipe_file(key, raw_validity)
    assert fs.cat_file(key) == raw_validity
    files["all-native-validity.json"] = {"bytes": len(raw_validity), "sha256": sha(raw_validity)}
    (out / "all-native-validity.json").write_bytes(raw_validity)
    freeze = {
        "schema_version": 1,
        "state": "REGIONAL_BYTES_AND_NATIVE128_VALIDITY_VERIFIED_NOT_TRAINING_APPROVAL",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "root": root,
        "files": files,
        "logical_packet_sha256": sha(logical_packet.read_bytes()),
        "relocation_code_sha256": sha(Path(__file__).read_bytes()),
        "old_rows": 767,
        "new_rows": 322,
        "combined_rows": 1089,
        "old_cache_files_byte_identical": 18,
        "new_cache_files_byte_identical": 27,
        "component_order": expected["component_order"],
        "sampling_weights": expected["sampling_weights"],
        "per_row_weights": expected["per_row_weights"],
        "seed": SEED,
        "feistel": True,
        "mixture_block_size": 2048,
        "all128_order_ID_token_mask_and_validity_exact": True,
        "all322_new_validity_exact": True,
        "first128": expected["first128"],
        "per_update": expected["per_update"],
        "history_new_rows": 0,
        "no_target_repair_replacement_regeneration": True,
        "training_approved": False,
    }
    raw = encoded(freeze)
    fs, key = filesystem_for(root + "/freeze.json")
    assert not fs.exists(key)
    fs.pipe_file(key, raw)
    assert fs.cat_file(key) == raw
    (out / "freeze.json").write_bytes(raw)
    print(
        json.dumps(
            {
                "freeze_sha256": sha(raw),
                "root": root,
                "files_readback": len(files),
                "first128": expected["first128"],
                "training_approved": False,
            },
            indent=2,
        )
    )
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "preview", "root", "out", "logical-packet"):
        parser.add_argument("--" + name, required=True, type=str if name == "root" else Path)
    parser.add_argument("--approved-logical-packet-sha256", required=True)
    args = parser.parse_args()
    run(args.old, args.preview, args.root, args.out, args.logical_packet, args.approved_logical_packet_sha256)


if __name__ == "__main__":
    main()
