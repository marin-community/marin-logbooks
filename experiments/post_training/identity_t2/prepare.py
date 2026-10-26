# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a CPU-only native input/share preview; not a freeze or submit command."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter
from itertools import batched
from pathlib import Path

import draccus
import haliax as hax
import jax
import numpy as np
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat, LmDatasetFormatBase
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheMetadata, SerialCacheWriter
from levanter.tokenizers import MarinTokenizer

SEQ_LEN = 8192
BATCH_SIZE = 8
SEED = 310
NEW_ROW_SAMPLING_WEIGHTS = {"identity_v2": 1.2, "tools": 2.0, "third_person": 1.0}
POST_CENSUS_SHA = "1fedcaa95c10544bc25f641aba475b541a8e5c8dbaba58479d897003444e5202"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha(array) -> str:
    return sha(np.asarray(array, dtype="<i4").tobytes())


def encoded_record(row: dict) -> tuple[dict, dict]:
    prefix = row["student_prompt_token_ids"]
    target = row["raw_target_token_ids"]
    assert not row["decision"]["rejection_reasons"] and target[-1] == 128009
    assert row["target_start"] == len(prefix) and row["valid_target_length"] == len(target)
    full = [*prefix, *target]
    if len(full) > SEQ_LEN or not prefix:
        raise ValueError("Captured target exceeds the unchanged native context or has no prefix")
    # Keep the cache variable-length. Native ChatDataset packing pads later and
    # supplies explicit segment=-1 for padding, including when PAD==EOS.
    tokens = np.asarray(full, dtype=np.int32)
    assistant_masks = np.zeros(len(full), dtype=np.int32)
    assistant_masks[len(prefix) :] = 1
    # Expected native causal mask is shifted once by ChatDataset, not here.
    expected_tokens = np.pad(tokens, (0, SEQ_LEN - len(full)))
    expected_weights = np.zeros(SEQ_LEN, dtype=np.int32)
    expected_weights[len(prefix) - 1 : len(full) - 1] = 1
    metadata = {
        "id": row["id"],
        "component": row["candidate"]["group"],
        "family": row["candidate"]["family"],
        "valid_input_length": len(full),
        "target_start": len(prefix),
        "valid_target_length": len(target),
        "supervised_tokens": len(target),
        "class_tokens": {
            "identity_v2" if row["candidate"]["group"] == "identity_v2" else row["candidate"]["group"]: len(target)
        },
        "tokens_sha256": array_sha(expected_tokens),
        "target_mask_sha256": array_sha(expected_weights),
        "raw_record_sha256": row["raw_record_sha256"],
    }
    return {"input_ids": tokens, "assistant_masks": assistant_masks}, metadata


class CapturedTargetProcessor(BatchProcessor[dict, dict]):
    """Emit original variable-length IDs and length-derived final-target masks."""

    def __call__(self, batch):
        return [encoded_record(row)[0] for row in batch]

    @property
    def output_exemplar(self):
        return {"input_ids": np.zeros(0, dtype=np.int32), "assistant_masks": np.zeros(0, dtype=np.int32)}

    @property
    def metadata(self):
        return {
            "type": "t2_captured_chat",
            "version": 1,
            "raw_ids_authoritative": True,
            "mask_source": "explicit_prefix_and_target_lengths",
            "padding": "native_chat_segments_minus1",
        }

    @property
    def num_cpus(self):
        return 1


@LmDatasetFormatBase.register_subclass("t2_captured_chat")
@dataclasses.dataclass(frozen=True)
class CapturedChatFormat(ChatLmDatasetFormat):
    """Use native ChatDataset/packing without retokenizing captured targets."""

    def build_preprocessor(self, tokenizer: MarinTokenizer, *, enforce_eos=True, enforce_bos=True):
        del tokenizer, enforce_eos, enforce_bos
        return CapturedTargetProcessor()


def check_validity(query_segments, key_segments, meta: dict) -> dict:
    query, key = np.asarray(query_segments), np.asarray(key_segments)
    expected = np.arange(SEQ_LEN) < meta["valid_input_length"]
    np.testing.assert_array_equal(query >= 0, expected)
    np.testing.assert_array_equal(key >= 0, expected)
    assert np.all(query[~expected] == -1) and np.all(key[~expected] == -1)
    assert len(np.unique(query[expected])) == len(np.unique(key[expected])) == 1
    return {
        "id": meta["id"],
        "valid_input_length": meta["valid_input_length"],
        "actual_valid_query_positions": int((query >= 0).sum()),
        "actual_valid_key_positions": int((key >= 0).sum()),
        "padding_segments_minus1": True,
    }


def summarize(occurrences: list[dict]) -> dict:
    classes = Counter()
    for row in occurrences:
        classes.update(row["class_tokens"])
    total = sum(classes.values())
    identity = sum(value for key, value in classes.items() if key.startswith("identity_"))
    return {
        "positions": len(occurrences),
        "unique_ids": len({row["id"] for row in occurrences}),
        "component_occurrences": dict(Counter(row["component"] for row in occurrences)),
        "class_tokens": dict(classes),
        "supervised_tokens": total,
        "identity_tokens": identity,
        "identity_token_share": identity / total,
        "tool_token_share": classes.get("tools", 0) / total,
    }


def run(old_root: Path, candidates: Path, out: Path) -> dict:
    if jax.default_backend() != "cpu":
        raise ValueError("Dataset preview is authorized for CPU only")
    assert sha(candidates.read_bytes()) == POST_CENSUS_SHA
    old_manifest = json.loads((old_root / "manifest.json").read_bytes())
    old_file_checks = {}
    for relative, info in old_manifest["cache_files"].items():
        path = old_root / relative
        assert sha(path.read_bytes()) == info["sha256"]
        old_file_checks[relative] = info["sha256"]
    for source_path, expected in old_manifest["source_files"].items():
        assert sha(Path(source_path).read_bytes()) == expected
    old_rows = json.loads((old_root / "native-row-fingerprints.json").read_bytes())
    assert len(old_rows) == 767
    rows = [json.loads(line) for line in candidates.read_bytes().splitlines()]
    assert len(rows) == 322
    assert Counter(row["candidate"]["group"] for row in rows) == {"tools": 156, "identity_v2": 112, "third_person": 54}
    data = draccus.decode(LmDataConfig, json.loads((old_root / "data-config.json").read_bytes()))
    assert data.permutation_type == "feistel" and data.mixture_block_size == 2048
    assert data.auto_build_caches is False and data.train_weights == {"identity": 264.0, "preservation": 547.0}
    assert list(data.components) == ["identity", "preservation"]
    tokenizer = data.the_tokenizer  # Native URL loader stages tokenizer allow-patterns only, never model payloads.
    assert sha(tokenizer.chat_template.encode()) == "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de"
    out.mkdir(parents=True, exist_ok=False)
    components = dict(data.components)
    train_weights = dict(data.train_weights)
    metadata = dict(old_rows)
    fingerprints = {(row["tokens_sha256"], row["target_mask_sha256"]): key for key, row in metadata.items()}
    assert len(fingerprints) == 767
    for group in ("identity_v2", "tools", "third_person"):
        subset = [row for row in rows if row["candidate"]["group"] == group]
        source = out / (group + ".jsonl")
        fmt = CapturedChatFormat(pack=False, slice_strategy="raise", mask_user_turns=True)
        processor = fmt.build_preprocessor(tokenizer)
        cache_root = out / "cache" / group
        with (
            source.open("x") as stream,
            SerialCacheWriter(
                str(cache_root / "train"),
                processor.output_exemplar,
                CacheMetadata(preprocessor_metadata=processor.metadata),
            ) as writer,
        ):
            for batch in batched(subset, 16):
                for row in batch:
                    _, meta = encoded_record(row)
                    key = (meta["tokens_sha256"], meta["target_mask_sha256"])
                    assert key not in fingerprints and row["id"] not in metadata
                    fingerprints[key] = row["id"]
                    metadata[row["id"]] = meta
                    stream.write(json.dumps(row) + "\n")
                writer.write_batch(processor(list(batch)))
        components[group] = DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[str(source)]), cache_dir=str(cache_root), format=fmt, pack=False
        )
        train_weights[group] = len(subset) * NEW_ROW_SAMPLING_WEIGHTS[group]
    assert train_weights == {
        "identity": 264.0,
        "preservation": 547.0,
        "identity_v2": 134.4,
        "tools": 312.0,
        "third_person": 54.0,
    }
    data = dataclasses.replace(data, components=components, train_weights=train_weights, cache_catalog=None)
    catalog = out / "cache-catalog.json"
    data.build_cache_catalog(str(catalog), splits=("train",))
    data = dataclasses.replace(data, cache_catalog=str(catalog))
    (out / "data-config.json").write_text(json.dumps(draccus.encode(data), indent=2) + "\n")
    (out / "native-row-fingerprints.json").write_text(json.dumps(metadata, indent=2) + "\n")
    new_ids = {row["id"] for row in rows}
    actual_validity = []
    native_sets = data.train_sets(
        hax.Axis("position", SEQ_LEN), initial_batch_size=BATCH_SIZE, key=jax.random.PRNGKey(SEED)
    )
    for group in ("identity_v2", "tools", "third_person"):
        group_size = sum(row["candidate"]["group"] == group for row in rows)
        native_set = native_sets[group].as_sync_dataset()
        for indices in batched(range(group_size), BATCH_SIZE):
            for example in native_set.get_batch(list(indices)):
                tokens, weights = np.asarray(example.tokens), np.asarray(example.loss_weight)
                key = (array_sha(tokens), array_sha(weights))
                meta = metadata[fingerprints[key]]
                assert meta["component"] == group
                assert int(weights.sum()) == meta["supervised_tokens"]
                query, key_segments = example.attn_mask.segment_ids
                actual_validity.append(check_validity(query, key_segments, meta))
    assert len(actual_validity) == 322 and {row["id"] for row in actual_validity} == new_ids
    validity_path = out / "native-validity-all-new.json"
    validity_path.write_text(json.dumps(actual_validity, indent=2) + "\n")
    dataset = data.train_set(
        hax.Axis("position", SEQ_LEN), BatchSchedule(BATCH_SIZE), key=jax.random.PRNGKey(SEED)
    ).as_sync_dataset()
    occurrences = []
    for positions in batched(range(128), BATCH_SIZE):
        examples = dataset.get_batch(list(positions))
        for position, example in zip(positions, examples, strict=True):
            tokens = np.asarray(example.tokens.array)
            weights = np.asarray(example.loss_weight.array)
            assert tokens.shape == weights.shape == (SEQ_LEN,)
            assert np.all((weights == 0) | (weights == 1))
            key = (array_sha(tokens), array_sha(weights))
            identifier = fingerprints[key]
            meta = metadata[identifier]
            assert int(weights.sum()) == meta["supervised_tokens"]
            if identifier in new_ids:
                assert not np.any(weights[: meta["target_start"] - 1])
                assert not np.any(weights[meta["valid_input_length"] - 1 :])
                query, key_segments = example.attn_mask.segment_ids
                check_validity(query.array, key_segments.array, meta)
            occurrences.append({"position": position, "update": position // BATCH_SIZE, **meta})
    summary = summarize(occurrences)
    updates = [summarize(list(batch)) for batch in batched(occurrences, BATCH_SIZE)]
    report = {
        "CPU_native_iterator_not_model_compute": True,
        "old_767_cache_files_unchanged": old_file_checks,
        "old_767_rows_unchanged": True,
        "new_rows": 322,
        "combined_rows": 1089,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "position_length": SEQ_LEN,
        "permutation_type": data.permutation_type,
        "mixture_block_size": data.mixture_block_size,
        "component_order": list(components),
        "sampling_weights": train_weights,
        "per_identity_row_sampling_weight": 1.2,
        "tool_row_sampling_weight": 2.0,
        "control_and_preservation_row_sampling_weight": 1.0,
        "pre_freeze_owner_tool_weight_amendment": "1.0 to 2.0; sampling only, never loss-mask scaling",
        "target_masks_from_explicit_prefix_and_target_lengths": True,
        "new_source_assistant_history_supervision": False,
        "native_new_row_format": "variable-length t2_captured_chat through native ChatDataset packing",
        "actual_segment_validity_checked_all_new_rows": len(actual_validity),
        "actual_validity_sha256": sha(validity_path.read_bytes()),
        "padding_segments_are_minus1_on_actual_examples": True,
        "first128": summary,
        "per_update": updates,
        "mean_update_identity_token_fraction": sum(row["identity_token_share"] for row in updates) / len(updates),
        "owner_escalation_above_8_percent": summary["identity_token_share"] > 0.08,
        "freeze_approved": False,
        "training_approved": False,
        "cache_catalog_sha256": sha(catalog.read_bytes()),
        "candidate_sha256": POST_CENSUS_SHA,
        "old_manifest_sha256": sha((old_root / "manifest.json").read_bytes()),
    }
    (out / "occurrences-128.jsonl").write_text("".join(json.dumps(row) + "\n" for row in occurrences))
    (out / "share-preview.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-root", "candidates", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.old_root, args.candidates, args.out)


if __name__ == "__main__":
    main()
