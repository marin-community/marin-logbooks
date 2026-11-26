# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnose what the V9 rl_lexical mark fires on (one-off evidence for the filter rule).

For a sample of rl_lexical-dropped rows (from the versioned shards' decisions
ledgers) this job decodes the row's unmasked span, re-runs the decontamination
mark against the RL-prompt bloom WITHOUT the boilerplate drop set, and prints
the exact paragraphs that cross the mark threshold together with the index
multiplicity of their matched hashes (how many RL prompts share each 13-gram).
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from levanter.store.jagged_array import JaggedArrayStore
from marin.datakit.decon import (
    DEFAULT_PARAGRAPH_DELIMITER,
    NGramConfig,
    _bloom_hash,
    _document_overlap_and_matches,
    _extract_features,
    _iter_paragraphs,
    _load_bloom,
)
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=2)


def sample_dropped_rows(store_dir: str, samples_per_shard: int) -> list[dict]:
    """Read the shard decisions ledgers and sample rl_lexical-dropped rows."""
    sampled = []
    shards_root = StoragePath(f"{store_dir}/shards")
    entries = sorted(str(entry) for entry in shards_root.ls())
    for shard in entries:
        decisions_path = StoragePath(f"{shard}/decisions.jsonl")
        if not decisions_path.isfile():
            continue
        rows = []
        for line in decisions_path.read_text().splitlines():
            record = json.loads(line)
            if record["stage"] == "rl_lexical":
                rows.append({"shard_root": shard, "row": record["row"], "reason": record["reason"]})
        if not rows:
            continue
        step = max(1, len(rows) // samples_per_shard)
        sampled.extend(rows[::step][:samples_per_shard])
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", required=True, help="A V9 versioned store's train/ dir")
    parser.add_argument("--bloom-manifest-uri", required=True)
    parser.add_argument("--bloom-manifest-sha256", required=True)
    parser.add_argument("--tokenizer-uri", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=2)
    parser.add_argument("--output", required=True, help="Local output path for the diagnostic JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()

    bloom_manifest = json.loads(read_pinned(args.bloom_manifest_uri, args.bloom_manifest_sha256).decode())
    bloom = _load_bloom(bloom_manifest["bloom"]["uri"])
    tokenizer = Tokenizer.from_str(read_pinned(args.tokenizer_uri, args.tokenizer_sha256).decode())

    work = Path(tempfile.mkdtemp(prefix="v9-diag-"))
    eval_ids_by_hash: dict[int, list[str]] = {}
    for part in bloom_manifest["index_parts"]:
        fs, key = filesystem_for(part["uri"])
        local = work / f"index-{Path(part['uri']).name}"
        local.write_bytes(fs.cat(key))
        table = pq.read_table(local, columns=["hash", "eval_id"])
        for value, eval_id in zip(table.column("hash").to_pylist(), table.column("eval_id").to_pylist(), strict=True):
            eval_ids_by_hash.setdefault(value, []).append(eval_id)

    sampled = sample_dropped_rows(args.store_dir, args.samples_per_shard)
    logger.info("sampled %d rl_lexical rows", len(sampled))
    findings = []
    for sample in sampled:
        job = json.loads(StoragePath(f"{sample['shard_root']}/receipt.json").read_text())["job"]
        part = f"{job['base_cache_dir'].rstrip('/')}/{job['base_shard_key']}"
        ids_store = JaggedArrayStore.open(f"{part}/input_ids", mode="r", dtype=np.int32)
        weights_store = JaggedArrayStore.open(f"{part}/loss_weights", mode="r", dtype=np.float32)
        ids = ids_store.get_batch_sync([sample["row"]])[0]
        weights = weights_store.get_batch_sync([sample["row"]])[0]
        text = tokenizer.decode(ids[weights == 0].tolist(), skip_special_tokens=False)
        score, matched = _document_overlap_and_matches(text, bloom, NGRAM)
        firing = []
        for paragraph in _iter_paragraphs(text, DEFAULT_PARAGRAPH_DELIMITER):
            paragraph_hashes = {_bloom_hash(f) for f in _extract_features(paragraph, NGRAM)}
            hits = [h for h in paragraph_hashes if h in bloom]
            if len(hits) >= 2 and len(hits) / max(1, len(paragraph_hashes)) >= 0.5:
                multiplicities = [len(eval_ids_by_hash.get(h, [])) for h in hits]
                firing.append(
                    {
                        "paragraph_chars": len(paragraph),
                        "ngram_features": len(paragraph_hashes),
                        "matched": len(hits),
                        "score": round(len(hits) / max(1, len(paragraph_hashes)), 3),
                        "hash_eval_id_multiplicity_min_max": [min(multiplicities), max(multiplicities)],
                        "head": paragraph[:300],
                    }
                )
        findings.append(
            {
                "row": sample["row"],
                "score": score,
                "matched": len(matched),
                "firing_paragraphs": firing,
                "user_span_head": text[:200],
            }
        )

    out = {"store_dir": args.store_dir, "samples": findings}
    Path(args.output).write_text(json.dumps(out, indent=1))
    fired = [f for f in findings if f["firing_paragraphs"]]
    print(json.dumps({"samples": len(findings), "with_firing_paragraphs": len(fired)}, indent=1))


if __name__ == "__main__":
    main()
