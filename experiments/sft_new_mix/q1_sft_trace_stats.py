# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-SFT-source trace-style statistics over the V7 release caches (Q1).

Question: which SFT sources in V7 teach long, recheck-heavy reasoning on the
Nemotron-style task families (competitive coding, STEM MCQ, puzzles)? For every
``sft/source/*`` component of the release this script samples rows from the
sealed token caches, measures the assistant span (the ``loss_weights`` mask of
the ``chat_assistant_only`` sources), and reports per source:

- mix share (``train_weights`` phase weights, identical across the three
  phases) and natural-population share;
- assistant-token length distribution (p50/p90/p99), a log2-binned length
  histogram, and short/mid/long mode shares — per the coordinator's bimodality
  check (the policy's response lengths are bimodal, so a source that taught the
  never-commit mode should show a heavy long mode itself);
- re-verify / restart phrase rates per 1,000 assistant tokens ("wait", the
  "let me recheck" family, "alternatively", and a broader recheck list);
- share of assistant spans with no final-answer marker, plus completeness
  signals (ends with EOT, ends mid-sentence, opens with a turn header).

Nemotron SFT stores are reported alongside the other reasoning stores so the
two groups can be compared directly. Runs in the cache region (all V7 shards
are on ``s3://marin-us-east-08a``); reads a contiguous seeded block of rows
from a spread of shards per source, decodes only the masked span with the
pinned release tokenizer, and writes a receipt plus per-source records.
"""

import argparse
import hashlib
import json
import logging
import math
import random
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).parent / "fixtures"
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
TARGET_ROWS_PER_SOURCE = 2000
TOKEN_BUDGET_PER_SOURCE = 80_000_000
MAX_SHARDS_PER_SOURCE = 16
PREVIEW_CHARS = 220
HISTOGRAM_BIN_EDGES = tuple(2**k for k in range(10, 19))
"""Log2 assistant-token histogram edges: [512, 1024, ..., 131072); counts land left-inclusive."""
LONG_MODE_TOKENS = 16_384
SHORT_MODE_TOKENS = 1_024

# Worker-process state, set by _init_worker.
_ENTRIES: dict[str, dict] = {}
_TOKENIZER: Tokenizer | None = None
_TARGET_ROWS = TARGET_ROWS_PER_SOURCE
_TOKEN_BUDGET = TOKEN_BUDGET_PER_SOURCE
_MAX_SHARDS = MAX_SHARDS_PER_SOURCE

# Sources whose text is synthetic model output; user-conversation sources are
# excluded from preview capture.
PREVIEW_FAMILIES = frozenset(
    {
        "nemotron_sft_v3",
        "open_swe_traces",
        "openthoughts4-code-glm-5.2-n4",
        "calibforge-relay",
        "glm-5.2-kernelgym-rollouts",
        "science-tool-use-conversations",
        "synthetic-misconceptions-conversations",
    }
)

# Primary recheck phrases from the brief, plus the broader re-verify/restart
# list; every pattern is case-insensitive on word boundaries.
RECHECK_PATTERNS = {
    "wait": re.compile(r"\bwait\b", re.I),
    "let_me_recheck_family": re.compile(
        r"\blet me (?:re-?check|check|verify|reconsider|re-?examine|re-?read|confirm|go back|see again)\b",
        re.I,
    ),
    "alternatively": re.compile(r"\balternatively\b", re.I),
    "hmm": re.compile(r"\bhm{2,}\b", re.I),
    "actually": re.compile(r"\bactually\b", re.I),
    "hold_on": re.compile(r"\bhold on\b", re.I),
    "second_thought": re.compile(r"\bon (?:my |the )?second thought\b", re.I),
    "restart": re.compile(
        r"\blet me (?:try again|start over|restart|approach this differently|think again)\b",
        re.I,
    ),
    "error_correction": re.compile(
        r"\b(?:I made an error|I made a mistake|that'?s wrong|that is wrong|this is wrong|I was wrong|incorrect)\b",
        re.I,
    ),
    "double_check": re.compile(r"\bdouble[- ]check\b", re.I),
}
RECHECK_PRIMARY = ("wait", "let_me_recheck_family", "alternatively")

BOXED = re.compile(r"\\boxed\{")
ANSWER_TAG = re.compile(r"</?answer>", re.I)
ANSWER_MARKER = re.compile(r"(?:\bfinal answer\b|\bthe answer is\b|^\s*(?:final answer|answer)\s*[:=])", re.I | re.M)
SPECIAL_TAIL = re.compile(r"<\|[^|<>]+\|>\s*$")
SPECIAL_HEAD = re.compile(r"^\s*<\|[^|<>]+\|>")


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def family_of(source: str) -> str:
    parts = source.split("/")
    return parts[2] if parts[0] == "sft" and len(parts) > 2 else parts[0]


def choose_shards(shard_rows: dict[str, int], max_shards: int, target_rows: int) -> tuple[list[str], int]:
    """Evenly spread shard picks over the source's shard list, with the block size."""
    keys = sorted(shard_rows)
    if not keys:
        raise ValueError("source has no finished shards")
    n_pick = min(len(keys), max_shards)
    if len(keys) > n_pick:
        step = len(keys) / n_pick
        picked = [keys[min(len(keys) - 1, int(i * step))] for i in range(n_pick)]
    else:
        picked = list(keys)
    block = max(1, math.ceil(target_rows / len(picked)))
    return picked, block


def block_start(rows_in_shard: int, block: int, rng: random.Random) -> tuple[int, int]:
    if rows_in_shard <= block:
        return 0, rows_in_shard
    start = rng.randrange(0, rows_in_shard - block + 1)
    return start, block


def analyze_rows(
    rows: list[tuple[int, int, str, list[str], list[str]]],
    previews_allowed: bool,
) -> dict:
    """Aggregate per-row measurements into the per-source record.

    Each row is ``(assistant_tokens, total_tokens, assistant_text,
    nonbinary_loss_values, preview_tails)``.
    """
    n_rows = len(rows)
    lengths = [n_assist for n_assist, _, _, _, _ in rows if n_assist > 0]
    total_lengths = [n_total for _, n_total, _, _, _ in rows]
    total_assist = sum(lengths)
    zero_assist = sum(1 for n_assist, _, _, _, _ in rows if n_assist == 0)
    per_1k = 1000.0 / total_assist if total_assist else 0.0

    histogram = [0] * (len(HISTOGRAM_BIN_EDGES) + 1)
    for n_assist in lengths:
        bin_index = len(HISTOGRAM_BIN_EDGES)
        for edge_index, edge in enumerate(HISTOGRAM_BIN_EDGES):
            if n_assist < edge:
                bin_index = edge_index
                break
        histogram[bin_index] += 1
    share_short = sum(1 for n in lengths if n < SHORT_MODE_TOKENS) / len(lengths) if lengths else 0.0
    share_long = sum(1 for n in lengths if n >= LONG_MODE_TOKENS) / len(lengths) if lengths else 0.0
    share_mid = 1.0 - share_short - share_long

    phrase_counts = {name: 0 for name in RECHECK_PATTERNS}
    rows_with_any = 0
    rows_with_primary = 0
    nonbinary_rows = 0
    nonbinary_values: set[str] = set()

    n_boxed = n_tag = n_marker = n_any_answer = 0
    n_ends_eot = n_ends_complete = n_ends_word = n_starts_header = 0
    n_ge_8k = n_ge_16k = 0
    n_start_think = n_end_think = 0

    longest_n = -1
    longest_tail = None
    median_tail = None

    for n_assist, _, text, nonbinary, tails in rows:
        counts = {name: len(pattern.findall(text)) for name, pattern in RECHECK_PATTERNS.items()}
        for name, count in counts.items():
            phrase_counts[name] += count
        rows_with_any += int(any(counts.values()))
        rows_with_primary += int(any(counts[name] for name in RECHECK_PRIMARY))
        has_boxed = bool(BOXED.search(text))
        has_tag = bool(ANSWER_TAG.search(text))
        has_marker = bool(ANSWER_MARKER.search(text))
        n_boxed += has_boxed
        n_tag += has_tag
        n_marker += has_marker
        n_any_answer += int(has_boxed or has_tag or has_marker)
        stripped = text.rstrip()
        if stripped:
            n_ends_eot += int(bool(SPECIAL_TAIL.search(stripped)))
            n_starts_header += int(bool(SPECIAL_HEAD.match(text)))
            last = stripped[-1]
            n_ends_complete += int(last in ".!?'\"`)}]:;")
            n_ends_word += int(last.isalnum())
        n_ge_8k += int(n_assist >= 8000)
        n_ge_16k += int(n_assist >= 16000)
        n_start_think += int("<|start_think|>" in text or "</think>" in text)
        n_end_think += int("<|end_think|>" in text or "</think>" in text)
        if nonbinary:
            nonbinary_rows += 1
            nonbinary_values.update(nonbinary)

        if previews_allowed and tails:
            if n_assist > longest_n:
                longest_n = n_assist
                longest_tail = tails[0]
            if median_tail is None:
                median_tail = tails[-1]

    record = {
        "rows_sampled": n_rows,
        "rows_zero_assistant": zero_assist,
        "assistant_tokens": {
            "total": total_assist,
            "mean": total_assist / len(lengths) if lengths else 0.0,
            "p50": percentile(lengths, 50),
            "p90": percentile(lengths, 90),
            "p99": percentile(lengths, 99),
            "max": max(lengths) if lengths else 0,
        },
        "assistant_token_histogram": {
            "edges_left_inclusive": list(HISTOGRAM_BIN_EDGES),
            "counts": histogram,
            "share_short_below_1k": share_short,
            "share_mid_1k_to_16k": share_mid,
            "share_long_at_or_above_16k": share_long,
        },
        "row_total_tokens_p50": percentile(total_lengths, 50),
        "share_assistant_ge_8k": n_ge_8k / n_rows if n_rows else 0.0,
        "share_assistant_ge_16k": n_ge_16k / n_rows if n_rows else 0.0,
        "recheck_counts": phrase_counts,
        "recheck_rate_per_1k_tokens": {name: phrase_counts[name] * per_1k for name in phrase_counts},
        "recheck_rate_primary_per_1k": sum(phrase_counts[name] for name in RECHECK_PRIMARY) * per_1k,
        "recheck_rate_all_per_1k": sum(phrase_counts.values()) * per_1k,
        "share_rows_with_any_recheck": rows_with_any / n_rows if n_rows else 0.0,
        "share_rows_with_primary_recheck": rows_with_primary / n_rows if n_rows else 0.0,
        "final_answer": {
            "share_boxed": n_boxed / n_rows if n_rows else 0.0,
            "share_answer_tag": n_tag / n_rows if n_rows else 0.0,
            "share_answer_marker": n_marker / n_rows if n_rows else 0.0,
            "share_no_final_answer": (n_rows - n_any_answer) / n_rows if n_rows else 0.0,
        },
        "completeness": {
            "share_ends_eot_like": n_ends_eot / n_rows if n_rows else 0.0,
            "share_ends_punctuation": n_ends_complete / n_rows if n_rows else 0.0,
            "share_ends_word": n_ends_word / n_rows if n_rows else 0.0,
            "share_starts_header": n_starts_header / n_rows if n_rows else 0.0,
        },
        "think_tag_structure": {
            "share_start_think": n_start_think / n_rows if n_rows else 0.0,
            "share_end_think": n_end_think / n_rows if n_rows else 0.0,
        },
        "loss_weight_anomaly_rows": nonbinary_rows,
        "nonbinary_loss_values": sorted(nonbinary_values)[:8],
    }
    if previews_allowed:
        record["previews"] = [tail for tail in (longest_tail, median_tail) if tail]
    return record


def sample_source(
    source: str,
    entry: dict,
    tokenizer: Tokenizer,
    target_rows: int,
    token_budget: int,
    max_shards: int,
) -> dict:
    """Sample rows from one source's sealed cache and measure the assistant span."""
    ledger = entry["ledger"]
    cache_dir = entry["cache_dir"].rstrip("/")
    shard_rows = ledger["shard_rows"]
    rng = random.Random(f"{SAMPLE_SEED}:{source}")
    shards, block = choose_shards(shard_rows, max_shards, target_rows)
    previews_allowed = family_of(source) in PREVIEW_FAMILIES

    rows_budget = min(target_rows, ledger["total_num_rows"])
    token_remaining = token_budget
    rows: list[tuple[int, int, str, list[str], list[str]]] = []
    shards_read = 0
    tokens_read = 0

    for shard in shards:
        if len(rows) >= rows_budget or token_remaining <= 0:
            break
        rows_in_shard = shard_rows[shard]
        start, count = block_start(rows_in_shard, block, rng)
        count = min(count, rows_budget - len(rows))
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
        shards_read += 1
        for ids, weights in zip(ids_batch, weights_batch, strict=True):
            mask = weights > 0
            n_assist = int(mask.sum())
            n_total = len(ids)
            tokens_read += n_total
            token_remaining -= n_total
            nonbinary: list[str] = []
            tails: list[str] = []
            text = ""
            if n_assist:
                text = tokenizer.decode(ids[mask].tolist(), skip_special_tokens=False)
                if previews_allowed:
                    tails = [text[-PREVIEW_CHARS:]]
                off_values = weights[(weights != 0.0) & (weights != 1.0)]
                if off_values.size:
                    nonbinary = [f"{value:.6g}" for value in np.unique(off_values)[:8]]
            rows.append((n_assist, n_total, text, nonbinary, tails))

    record = analyze_rows(rows, previews_allowed)
    record.update(
        {
            "source": source,
            "sample": {
                "shards_in_source": len(shard_rows),
                "shards_read": shards_read,
                "rows_in_source": ledger["total_num_rows"],
                "rows_budget": rows_budget,
                "tokens_read": tokens_read,
                "block_rows_per_shard": block,
                "seed_key": f"{SAMPLE_SEED}:{source}",
            },
        }
    )
    return record


def _init_worker(
    entries: dict[str, dict],
    tokenizer_json: str,
    target_rows: int,
    token_budget: int,
    max_shards: int,
) -> None:
    """Worker-side state: the sampled sources' catalog entries and the tokenizer."""
    global _ENTRIES, _TOKENIZER, _TARGET_ROWS, _TOKEN_BUDGET, _MAX_SHARDS
    _ENTRIES = entries
    _TOKENIZER = Tokenizer.from_str(tokenizer_json)
    _TARGET_ROWS = target_rows
    _TOKEN_BUDGET = token_budget
    _MAX_SHARDS = max_shards


def _run_one(source: str) -> dict:
    return sample_source(source, _ENTRIES[source], _TOKENIZER, _TARGET_ROWS, _TOKEN_BUDGET, _MAX_SHARDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, help="S3 prefix for the receipt and per-source records")
    parser.add_argument("--output", default=None, help="Local staging directory (default: a fresh temp dir)")
    parser.add_argument("--target-rows", type=int, default=TARGET_ROWS_PER_SOURCE)
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET_PER_SOURCE)
    parser.add_argument("--max-shards", type=int, default=MAX_SHARDS_PER_SOURCE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-sources", help="Comma-separated source-name substring filters (smoke runs)")
    parser.add_argument("--no-upload", action="store_true", help="Stage locally without writing to S3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    configure_storage()
    catalog = json.loads(read_pinned(CATALOG_URI, CATALOG_SHA256).decode())["splits"]["train"]
    data = json.loads(DATA_FIXTURE.read_text())
    weights = dict(data["train_weights"][0][1])
    phases_equal = all(dict(phase[1]) == weights for phase in data["train_weights"])

    sources = sorted(name for name in catalog if name.startswith("sft/source/"))
    if args.limit_sources:
        needles = [needle for needle in args.limit_sources.split(",") if needle]
        sources = [name for name in sources if any(needle in name for needle in needles)]
    if not sources:
        raise ValueError("no sft sources matched")
    total_tokens = sum(catalog[name]["ledger"]["field_counts"]["input_ids"] for name in catalog)

    tokenizer = Tokenizer.from_str(read_pinned(TOKENIZER_URI, TOKENIZER_SHA256).decode())
    entries = {source: catalog[source] for source in sources}

    records: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(entries, tokenizer.to_str(), args.target_rows, args.token_budget, args.max_shards),
    ) as pool:
        for source, record in zip(sources, pool.map(_run_one, sources), strict=True):
            records[source] = record
            logger.info(
                "%s: %d rows, p50 %.0f, recheck/1k %.2f",
                source,
                record["rows_sampled"],
                record["assistant_tokens"]["p50"],
                record["recheck_rate_all_per_1k"],
            )

    def mix_row(source: str) -> dict:
        tokens = catalog[source]["ledger"]["field_counts"]["input_ids"]
        return {
            "train_weight_phase0": weights[source],
            "phases_identical": phases_equal,
            "rows": catalog[source]["ledger"]["total_num_rows"],
            "tokens": tokens,
            "natural_token_share": tokens / total_tokens,
            "loss_mode": catalog[source]["ledger"]["metadata"]["preprocessor_metadata"]["loss_mode"],
        }

    groups: dict[str, dict] = {}
    for source, record in records.items():
        group = "nemotron_sft_v3" if family_of(source) == "nemotron_sft_v3" else "other_sft"
        weight = weights[source]
        stat = groups.setdefault(
            group,
            {
                "sources": 0,
                "train_weight_sum": 0.0,
                "tokens": 0,
                "p50_weighted": 0.0,
                "recheck_rate_all_per_1k_weighted": 0.0,
                "no_final_answer_weighted": 0.0,
            },
        )
        stat["sources"] += 1
        stat["train_weight_sum"] += weight
        stat["tokens"] += catalog[source]["ledger"]["field_counts"]["input_ids"]
        stat["p50_weighted"] += weight * record["assistant_tokens"]["p50"]
        stat["recheck_rate_all_per_1k_weighted"] += weight * record["recheck_rate_all_per_1k"]
        stat["no_final_answer_weighted"] += weight * record["final_answer"]["share_no_final_answer"]

    receipt = {
        "status": "COMPLETE",
        "question": "Which V7 SFT sources teach long/recheck-heavy reasoning on Nemotron-style task types (Q1)",
        "pins": {
            "catalog_uri": CATALOG_URI,
            "catalog_sha256": CATALOG_SHA256,
            "tokenizer_uri": TOKENIZER_URI,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "data_fixture_sha256": hashlib.sha256(DATA_FIXTURE.read_bytes()).hexdigest(),
            "sample_seed": SAMPLE_SEED,
        },
        "sample_design": {
            "target_rows_per_source": args.target_rows,
            "token_budget_per_source": args.token_budget,
            "max_shards_per_source": args.max_shards,
            "row_sampling": "seeded contiguous block per evenly-spread shard",
            "assistant_span": "loss_weights > 0 mask of the sealed chat_assistant_only caches",
        },
        "sources": {source: {"mix": mix_row(source), "measured": record} for source, record in sorted(records.items())},
        "groups": groups,
        "notes": (
            "Assistant-token counts come from the loss_weights mask; the seal manifests record field lengths, "
            "not masked totals, so these are the first direct per-row assistant measurements for the release. "
            "share_no_final_answer = no \\boxed, no <answer> tag, and no final-answer marker phrase; read agent "
            "and code families alongside share_ends_punctuation. Previews are captured only for synthetic "
            "model-output families."
        ),
    }

    staging = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="q1-sft-trace-stats-"))
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    for source, record in records.items():
        safe = source.replace("/", "__")
        (staging / f"{safe}.json").write_text(json.dumps(record, indent=2, sort_keys=True))

    if not args.no_upload:
        destination = StoragePath(args.destination)
        (destination / "receipt.json").write_text((staging / "receipt.json").read_text())
        for source in records:
            safe = source.replace("/", "__")
            (destination / f"sources/{safe}.json").write_text((staging / f"{safe}.json").read_text())

    logger.info("staged records under %s", staging)
    summary = {
        source: {
            "weight": weights[source],
            "p50": record["assistant_tokens"]["p50"],
            "p99": record["assistant_tokens"]["p99"],
            "recheck_all_per_1k": record["recheck_rate_all_per_1k"],
            "no_final_answer": record["final_answer"]["share_no_final_answer"],
        }
        for source, record in sorted(records.items())
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
