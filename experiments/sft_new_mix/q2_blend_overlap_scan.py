# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan V7 SFT caches for overlap with the RL blend prompts (Q2).

Question: are the Nemotron RL blend prompts — or near-duplicates from the same
generators — present in the Nemotron SFT stores, so that SFT taught a long-trace
style for exactly these prompt families?

The scan reuses the lexical-decontamination feature extractor
(``marin.datakit.decon``): word 13-grams, stride 0, with the Atlas cluster-D
alphabetic filter. The roles are reversed relative to the standard decontam
scan — the feature set is built from the 380 frozen sweep prompts (80 eval
blend components plus 300 train prompts across seven RL sources, shipped as
``q2_blend_prompts.json``), and the sampled SFT rows are the mark side. At
this scale the prompt feature set fits in memory, so membership is an exact
string set rather than a bloom: every reported hit is a verbatim 13-word
overlap, with no false positives.

Features shared by many sampled prompts (GENERIC_FEATURE_PROMPTS or more) are
boilerplate — instruction preambles and response-format templates — and are
tracked separately as convention sharing. Content overlap is per (row, prompt):
a row contains a prompt's content when at least OVERLAP_MIN_MATCHED distinct
features that appear in only that prompt (within the sample) occur in the row
and cover at least OVERLAP_THRESHOLD of that prompt's sample-specific features.

For every ``sft/source/*`` component the job samples rows (seeded contiguous
blocks on evenly spread shards, as in Q1), decodes only the ``loss_weights ==
0`` span (system + user turns; the assistant span is excluded by construction),
and reports per-source convention-sharing rates, content-match rates, and the
prompt attributions.
"""

import argparse
import hashlib
import json
import logging
import math
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from levanter.store.jagged_array import JaggedArrayStore
from marin.datakit.decon import _extract_ngrams
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).parent / "fixtures"
PROMPTS_FILE = Path(__file__).parent / "q2_blend_prompts.json"
DATA_FIXTURE = FIXTURES / "exp3_v7_data.json"

CATALOG_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.18-v7-option-g" "/release-v7/cache_catalog.json"
)
CATALOG_SHA256 = "a734403460002500ceac50fdcb6a43465c70fd57f7d05cf4a95b137dd330ca9c"
TOKENIZER_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/scripted-v3/tokenizer.json"
)
TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"

SAMPLE_SEED = 20261122
NGRAM_LENGTH = 13
"""The Atlas decontamination n-gram length (word tokens, stride 0)."""
GENERIC_FEATURE_PROMPTS = 3
"""A feature present in >= this many sampled prompts is boilerplate."""
OVERLAP_MIN_MATCHED = 8
"""Minimum distinct specific features of one prompt inside a row before overlap counts."""
OVERLAP_THRESHOLD = 0.2
"""Fraction of a prompt's specific features present in a row for a content match."""
ROWS_PER_SOURCE = 15_000
TOKEN_BUDGET_PER_SOURCE = 150_000_000
MAX_SHARDS_PER_SOURCE = 16
MAX_EXAMPLES_PER_SOURCE = 50

# Worker-process state, set by _init_worker.
_SPECIFIC_PROMPTS: dict[str, list[str]] = {}
"""specific feature -> prompt identities carrying it."""
_PROMPT_SPECIFIC_COUNT: dict[str, int] = {}
"""prompt identity -> |specific features| (content-overlap denominator)."""
_GENERIC_FEATURES: frozenset[str] = frozenset()
_TOKENIZER: Tokenizer | None = None
_TARGET_ROWS = ROWS_PER_SOURCE
_TOKEN_BUDGET = TOKEN_BUDGET_PER_SOURCE
_MAX_SHARDS = MAX_SHARDS_PER_SOURCE
_ENTRIES: dict[str, dict] = {}


def prompt_features(user_text: str) -> set[str]:
    """Matchable 13-gram features of one prompt (short prompts as one guarded feature)."""
    tokens = user_text.split()
    if len(tokens) < NGRAM_LENGTH:
        if len(tokens) >= 3 and any(map(str.isalpha, user_text)):
            return {" ".join(tokens)}
        return set()
    return set(_extract_ngrams(user_text, NGRAM_LENGTH, 0))


def choose_shards(shard_rows: dict[str, int], max_shards: int, target_rows: int) -> tuple[list[str], int]:
    keys = sorted(shard_rows)
    n_pick = min(len(keys), max_shards)
    if len(keys) > n_pick:
        step = len(keys) / n_pick
        picked = [keys[min(len(keys) - 1, int(i * step))] for i in range(n_pick)]
    else:
        picked = list(keys)
    block = max(1, math.ceil(target_rows / len(picked)))
    return picked, block


def scan_source(source: str) -> dict:
    """Probe sampled rows' unmasked spans against the prompt feature sets."""
    entry = _ENTRIES[source]
    ledger = entry["ledger"]
    cache_dir = entry["cache_dir"].rstrip("/")
    shard_rows = ledger["shard_rows"]
    rng = random.Random(f"{SAMPLE_SEED}:q2:{source}")
    shards, block = choose_shards(shard_rows, _MAX_SHARDS, _TARGET_ROWS)

    rows_scanned = 0
    rows_with_generic = 0
    rows_with_specific = 0
    rows_with_content_match = 0
    generic_hits = 0
    specific_hits = 0
    token_remaining = _TOKEN_BUDGET
    matched_generic: dict[str, int] = {}
    matched_specific: dict[str, int] = {}
    overlap_histogram = {"below_0.05": 0, "0.05_to_0.2": 0, "at_or_above_0.2": 0}
    prompt_row_matches: dict[str, int] = {}
    examples: list[dict] = []

    for shard in shards:
        if rows_scanned >= _TARGET_ROWS or token_remaining <= 0:
            break
        rows_in_shard = shard_rows[shard]
        start = 0 if rows_in_shard <= block else rng.randrange(0, rows_in_shard - block + 1)
        count = min(block, rows_in_shard - start, _TARGET_ROWS - rows_scanned)
        # Respect the token budget with the source's mean row length.
        mean_row = max(1, ledger["field_counts"]["input_ids"] // max(1, ledger["total_num_rows"]))
        count = min(count, max(1, token_remaining // mean_row))
        if count <= 0:
            continue
        part = f"{cache_dir}/{shard}"
        ids_store = JaggedArrayStore.open(f"{part}/input_ids", mode="r", dtype=np.int32)
        weights_store = JaggedArrayStore.open(f"{part}/loss_weights", mode="r", dtype=np.float32)
        ids_batch = ids_store.get_batch_sync(range(start, start + count))
        weights_batch = weights_store.get_batch_sync(range(start, start + count))
        for row_index, (ids, weights) in enumerate(zip(ids_batch, weights_batch, strict=True)):
            rows_scanned += 1
            token_remaining -= len(ids)
            unmasked = ids[weights == 0]
            if not len(unmasked):
                continue
            text = _TOKENIZER.decode(unmasked.tolist(), skip_special_tokens=False)
            row_generic: set[str] = set()
            row_specific: set[str] = set()
            for feature in _extract_ngrams(text, NGRAM_LENGTH, 0):
                if feature in _SPECIFIC_PROMPTS:
                    row_specific.add(feature)
                elif feature in _GENERIC_FEATURES:
                    row_generic.add(feature)
            if not row_generic and not row_specific:
                continue
            rows_with_generic += bool(row_generic)
            rows_with_specific += bool(row_specific)
            generic_hits += len(row_generic)
            specific_hits += len(row_specific)
            for feature in row_generic:
                matched_generic[feature] = matched_generic.get(feature, 0) + 1
            for feature in row_specific:
                matched_specific[feature] = matched_specific.get(feature, 0) + 1

            candidate_prompts = {prompt for feature in row_specific for prompt in _SPECIFIC_PROMPTS[feature]}
            best: tuple[str, float, int] | None = None
            for prompt in candidate_prompts:
                matched = sum(1 for feature in row_specific if prompt in _SPECIFIC_PROMPTS[feature])
                if matched < 3:
                    continue
                overlap = matched / _PROMPT_SPECIFIC_COUNT[prompt]
                if overlap < 0.05:
                    overlap_histogram["below_0.05"] += 1
                elif overlap < OVERLAP_THRESHOLD:
                    overlap_histogram["0.05_to_0.2"] += 1
                else:
                    overlap_histogram["at_or_above_0.2"] += 1
                if matched >= OVERLAP_MIN_MATCHED and overlap >= OVERLAP_THRESHOLD:
                    prompt_row_matches[prompt] = prompt_row_matches.get(prompt, 0) + 1
                    if best is None or overlap > best[1]:
                        best = (prompt, overlap, matched)
            if best is not None:
                rows_with_content_match += 1
                if len(examples) < MAX_EXAMPLES_PER_SOURCE:
                    prompt, overlap, matched = best
                    examples.append(
                        {
                            "shard": shard,
                            "row": start + row_index,
                            "prompt": prompt,
                            "overlap": round(overlap, 3),
                            "matched_features": matched,
                            "sample_features": sorted(row_specific)[:3],
                            "user_span_tokens": len(unmasked),
                        }
                    )

    return {
        "source": source,
        "rows_scanned": rows_scanned,
        "rows_with_generic_hits": rows_with_generic,
        "rows_with_specific_hits": rows_with_specific,
        "rows_with_content_match": rows_with_content_match,
        "generic_feature_hits": generic_hits,
        "specific_feature_hits": specific_hits,
        "distinct_specific_features": len(matched_specific),
        "overlap_histogram": overlap_histogram,
        "prompt_row_matches": prompt_row_matches,
        "top_generic_features": sorted(matched_generic.items(), key=lambda kv: -kv[1])[:20],
        "top_specific_features": sorted(matched_specific.items(), key=lambda kv: -kv[1])[:20],
        "examples": examples,
        "sample": {
            "shards_in_source": len(shard_rows),
            "block_rows_per_shard": block,
            "tokens_read": _TOKEN_BUDGET - token_remaining,
            "seed": f"{SAMPLE_SEED}:q2:{source}",
        },
    }


def _init_worker(
    entries: dict[str, dict],
    specific_prompts: dict[str, list[str]],
    prompt_specific_count: dict[str, int],
    generic: frozenset[str],
    tokenizer_json: str,
    rows_target: int,
    token_budget: int,
    max_shards: int,
) -> None:
    global _ENTRIES, _SPECIFIC_PROMPTS, _PROMPT_SPECIFIC_COUNT, _GENERIC_FEATURES
    global _TOKENIZER, _TARGET_ROWS, _TOKEN_BUDGET, _MAX_SHARDS
    _ENTRIES = entries
    _SPECIFIC_PROMPTS = specific_prompts
    _PROMPT_SPECIFIC_COUNT = prompt_specific_count
    _GENERIC_FEATURES = generic
    _TOKENIZER = Tokenizer.from_str(tokenizer_json)
    _TARGET_ROWS = rows_target
    _TOKEN_BUDGET = token_budget
    _MAX_SHARDS = max_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, help="S3 prefix for the receipt")
    parser.add_argument("--output", default=None, help="Local staging directory (default: a fresh temp dir)")
    parser.add_argument("--rows-per-source", type=int, default=ROWS_PER_SOURCE)
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET_PER_SOURCE)
    parser.add_argument("--max-shards", type=int, default=MAX_SHARDS_PER_SOURCE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-sources", help="Comma-separated source-name substring filters (smoke runs)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    configure_storage()
    catalog = json.loads(read_pinned(CATALOG_URI, CATALOG_SHA256).decode())["splits"]["train"]
    data = json.loads(DATA_FIXTURE.read_text())
    weights = dict(data["train_weights"][0][1])

    prompts = json.loads(PROMPTS_FILE.read_text())
    if not prompts:
        raise ValueError("prompt file is empty")
    feature_prompts: dict[str, list[str]] = {}
    for prompt in prompts:
        for feature in prompt_features(prompt["user_text"]):
            feature_prompts.setdefault(feature, []).append(prompt["identity"])
    specific_prompts = {f: ids for f, ids in feature_prompts.items() if len(ids) < GENERIC_FEATURE_PROMPTS}
    generic = frozenset(f for f, ids in feature_prompts.items() if len(ids) >= GENERIC_FEATURE_PROMPTS)
    prompt_specific_count = {
        prompt["identity"]: len(prompt_features(prompt["user_text"]) & set(specific_prompts)) for prompt in prompts
    }
    logger.info(
        "prompt features: %d total, %d specific, %d generic (shared by >=%d sampled prompts)",
        len(feature_prompts),
        len(specific_prompts),
        len(generic),
        GENERIC_FEATURE_PROMPTS,
    )

    sources = sorted(name for name in catalog if name.startswith("sft/source/"))
    if args.limit_sources:
        needles = [needle for needle in args.limit_sources.split(",") if needle]
        sources = [name for name in sources if any(needle in name for needle in needles)]
    if not sources:
        raise ValueError("no sft sources matched")
    tokenizer = Tokenizer.from_str(read_pinned(TOKENIZER_URI, TOKENIZER_SHA256).decode())
    entries = {source: catalog[source] for source in sources}

    records: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(
            entries,
            specific_prompts,
            prompt_specific_count,
            generic,
            tokenizer.to_str(),
            args.rows_per_source,
            args.token_budget,
            args.max_shards,
        ),
    ) as pool:
        for source, record in zip(sources, pool.map(scan_source, sources), strict=True):
            records[source] = record
            logger.info(
                "%s: %d rows, %d generic-hit, %d specific-hit, %d content-match",
                source,
                record["rows_scanned"],
                record["rows_with_generic_hits"],
                record["rows_with_specific_hits"],
                record["rows_with_content_match"],
            )

    labels = {prompt["identity"]: prompt["label"] for prompt in prompts}
    prompt_summary: dict[str, dict] = {}
    for source, record in records.items():
        for identity, rows in record["prompt_row_matches"].items():
            summary = prompt_summary.setdefault(
                identity, {"identity": identity, "label": labels[identity], "sources": {}}
            )
            summary["sources"][source] = rows

    receipt = {
        "status": "COMPLETE",
        "question": "Are the RL blend prompts (or same-generator near-duplicates) present in the V7 SFT stores (Q2)",
        "pins": {
            "catalog_uri": CATALOG_URI,
            "catalog_sha256": CATALOG_SHA256,
            "tokenizer_uri": TOKENIZER_URI,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "prompts_sha256": hashlib.sha256(PROMPTS_FILE.read_bytes()).hexdigest(),
            "prompts": len(prompts),
            "prompt_features": len(feature_prompts),
            "prompt_features_specific": len(specific_prompts),
            "prompt_features_generic": len(generic),
            "sample_seed": SAMPLE_SEED,
        },
        "method": {
            "ngram": f"{NGRAM_LENGTH} word tokens, stride 0, alphabetic filter (marin.datakit.decon)",
            "feature_match": "exact string membership (no false positives)",
            "generic_features": (
                f"13-grams present in >= {GENERIC_FEATURE_PROMPTS} sampled prompts (preambles, templates)"
            ),
            "mark_side": "loss_weights == 0 span (system + user turns) of sampled rows",
            "content_match": (
                f">= {OVERLAP_MIN_MATCHED} distinct specific features of one prompt in a row and >= "
                f"{OVERLAP_THRESHOLD:.0%} of that prompt's specific features"
            ),
            "rows_per_source": args.rows_per_source,
            "caveat": (
                "The 380 prompts are the sweep's frozen sample of the RL blends, not the full blend; "
                "hit rates estimate presence within the sampled SFT rows, not a full-population dedup. "
                "Generic-feature hit rates measure convention sharing (same preambles/templates), "
                "not identical tasks."
            ),
        },
        "sources": {
            source: {
                "train_weight_phase0": weights[source],
                "rows_in_source": catalog[source]["ledger"]["total_num_rows"],
                "scanned": {
                    key: record[key]
                    for key in (
                        "rows_scanned",
                        "rows_with_generic_hits",
                        "rows_with_specific_hits",
                        "rows_with_content_match",
                        "generic_feature_hits",
                        "specific_feature_hits",
                        "distinct_specific_features",
                        "overlap_histogram",
                        "sample",
                    )
                },
                "generic_hit_rate": record["rows_with_generic_hits"] / max(1, record["rows_scanned"]),
                "specific_hit_rate": record["rows_with_specific_hits"] / max(1, record["rows_scanned"]),
                "content_match_rate": record["rows_with_content_match"] / max(1, record["rows_scanned"]),
                "prompt_row_matches": record["prompt_row_matches"],
                "top_generic_features": record["top_generic_features"],
                "top_specific_features": record["top_specific_features"],
                "examples": record["examples"],
            }
            for source, record in sorted(records.items())
        },
        "prompt_summary": prompt_summary,
    }

    staging = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="q2-blend-overlap-"))
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    destination = StoragePath(args.destination)
    (destination / "receipt.json").write_text((staging / "receipt.json").read_text())

    logger.info("staged receipt under %s", staging)
    print(
        json.dumps(
            {
                source: {
                    "rows_scanned": record["rows_scanned"],
                    "rows_with_generic_hits": record["rows_with_generic_hits"],
                    "rows_with_specific_hits": record["rows_with_specific_hits"],
                    "rows_with_content_match": record["rows_with_content_match"],
                    "content_match_rate_pct": round(
                        100.0 * record["rows_with_content_match"] / max(1, record["rows_scanned"]), 4
                    ),
                }
                for source, record in sorted(records.items())
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
