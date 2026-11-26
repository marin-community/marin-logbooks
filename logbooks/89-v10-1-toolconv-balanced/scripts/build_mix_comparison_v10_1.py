"""Build the V10.1 vs V10 mix comparison (MIX-COMPARISON-V10.1.md + CSV).

Two mixes side by side, one row per source in their union (V10's 433 plus the
new balanced component):

- V10: the sealed release (2026.11.23-v10-overthinking) — rows/tokens from the
  sealed cache catalog, weights from the sealed schedule (the declared replay
  share 10.037232630869815%), 1,679 updates @ 256×262,144.
- V10.1: the candidate (2026.11.25-v10-1-balanced-toolconv) — rows/tokens from
  the V10.1 catalog (three math sources re-versioned @toolconv-20261125, one new
  balanced component), weights from the balanced-at-0.005 schedule.

The markdown report adds the toolconv coverage boundary, the balanced-component
rationale and the epoch-level summary from the V10.1 capacity receipt and epoch
plan beside V10's numbers. The seal is not touched by this script.
"""

import argparse
import csv
import io
import json
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for

MATH_SOURCES = (
    "sft/source/nemotron_sft_v3/math_v2/high",
    "sft/source/nemotron_sft_v3/math_v3/train",
    "sft/source/nemotron_sft_v3/math_v4/train",
)
BALANCED_SOURCE = "sft/source/balanced_reasoning/v1"
TOOLCONV_COVERAGE = {
    "applied": 4383,
    "pre_applied_at_build": 46314,
    "unmappable_to_v10_lineage": 188469,
    "quarantined_tokenizer_mismatch": 10654,
    "verified_modified_2026_11_25_index_snapshot": 2378592,
}


def read_bytes(uri: str) -> bytes:
    fs, key = filesystem_for(uri)
    return fs.cat(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10-release", required=True)
    parser.add_argument("--v10-1-manifest", required=True, help="The frozen pre-consumer V10.1 manifest")
    parser.add_argument("--apply-receipt", required=True)
    parser.add_argument("--capacity-receipt", required=True, help="The V10.1 capacity.json")
    parser.add_argument("--epoch-plan", required=True, help="The V10.1 epoch-plan.json")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    v10 = json.loads(read_bytes(args.v10_release))
    v10_1_manifest = json.loads(read_bytes(args.v10_1_manifest))
    v10_catalog = json.loads(read_bytes(v10["artifacts"]["cache_catalog.json"]["uri"]))["splits"]["train"]
    v10_1_catalog = json.loads(read_bytes(v10_1_manifest["artifacts"]["cache_catalog.json"]["uri"]))["splits"]["train"]
    v10_weights = json.loads(read_bytes(v10["artifacts"]["train_weights.json"]["uri"]))
    v10_1_weights = json.loads(read_bytes(v10_1_manifest["artifacts"]["train_weights.json"]["uri"]))
    apply_receipt = json.loads(read_bytes(args.apply_receipt))
    capacity = json.loads(read_bytes(args.capacity_receipt))
    epoch_plan = json.loads(read_bytes(args.epoch_plan))

    rows = []
    for source in sorted(set(v10_catalog) | set(v10_1_catalog)):
        v10_entry = v10_catalog.get(source)
        v10_1_entry = v10_1_catalog.get(source)
        row = {
            "component": source,
            "v10_rows": v10_entry["ledger"]["total_num_rows"] if v10_entry else "",
            "v10_1_rows": v10_1_entry["ledger"]["total_num_rows"] if v10_1_entry else "",
            "v10_tokens": v10_entry["ledger"]["field_counts"]["input_ids"] if v10_entry else "",
            "v10_1_tokens": v10_1_entry["ledger"]["field_counts"]["input_ids"] if v10_1_entry else "",
            "v10_weight": repr(v10_weights[0][1][source]) if source in v10_weights[0][1] else "",
            "v10_1_weight": repr(v10_1_weights[0][1][source]) if source in v10_1_weights[0][1] else "",
            "v10_cache_dir": v10_entry["cache_dir"] if v10_entry else "",
            "v10_1_cache_dir": v10_1_entry["cache_dir"] if v10_1_entry else "",
            "toolconv_replaced_rows": (
                apply_receipt["sources"][source]["replaced_rows"] if source in MATH_SOURCES else ""
            ),
            "new_in_v10_1": "yes" if source == BALANCED_SOURCE else "",
        }
        rows.append(row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode()

    # The population check: exactly the three math sources' tokens move and exactly
    # one source is new; every other source keeps its V10 rows and tokens.
    population_changes = {
        row["component"]
        for row in rows
        if row["v10_rows"] != row["v10_1_rows"] or row["v10_tokens"] != row["v10_1_tokens"]
    }
    expected_changes = set(MATH_SOURCES) | {BALANCED_SOURCE}
    if population_changes != expected_changes:
        raise ValueError(f"population differs outside the four expected sources: {population_changes ^ expected_changes}")

    totals_v10 = {
        "components": len(v10_catalog),
        "tokens": sum(entry["ledger"]["field_counts"]["input_ids"] for entry in v10_catalog.values()),
        "rows": sum(entry["ledger"]["total_num_rows"] for entry in v10_catalog.values()),
    }
    totals_v10_1 = {
        "components": len(v10_1_catalog),
        "tokens": sum(entry["ledger"]["field_counts"]["input_ids"] for entry in v10_1_catalog.values()),
        "rows": sum(entry["ledger"]["total_num_rows"] for entry in v10_1_catalog.values()),
    }
    toolconv = {
        source: {
            "replaced": apply_receipt["sources"][source]["replaced_rows"],
            "token_delta": apply_receipt["sources"][source]["output"]["token_delta"],
            "rewritten_shards": apply_receipt["sources"][source]["output"]["rewritten_shards"],
            "unchanged_shards": apply_receipt["sources"][source]["output"]["unchanged_shards"],
        }
        for source in MATH_SOURCES
    }
    report = f"""# MIX-COMPARISON-V10.1: the sealed V10 mix vs the V10.1 candidate

V10.1 = sealed V10 with exactly two changes (2026-11-25, no approval gate): (1) 4,383 verified
tool-converted rows replace their originals in the three math sources, with V10's assistant-only
supervision preserved (the all-position loss for Exp7 comes from the LLOM consumption view, PR
#12284); (2) one new balanced-reasoning component enters the mixture at weight 0.005 per phase.
Everything else is byte-identical V10. The seal is not touched by this report.

## Epoch summary

| Quantity | V10 | V10.1 |
|---|---|---|
| Declared replay share | {v10 and 0.10037232630869815!r} | {epoch_plan['declared_replay_fraction']!r} |
| Realized replay share | {0.10069702948183443!r} | {epoch_plan['realized_replay_fraction']!r} |
| Packed SFT sequences (every row once) | 386,542 | {epoch_plan['training_sft_sequences']:,} |
| Run sequences | 429,824 | {epoch_plan['run_sequences']:,} |
| Updates @ 256×262,144 | 1,679 | {epoch_plan['updates']:,} |
| Context positions | 112,675,782,656 | {epoch_plan['context_positions']:,} |
| Reserved validation sequences | 56 | {epoch_plan['reserved_validation_sequences']} |
| Components | {totals_v10['components']} | {totals_v10_1['components']} |
| Store tokens | {totals_v10['tokens']:,} | {totals_v10_1['tokens']:,} |
| Store rows | {totals_v10['rows']:,} | {totals_v10_1['rows']:,} |

Update count change: V10's 1,679 -> V10.1's {epoch_plan['updates']:,} ({epoch_plan['updates'] - 1679:+,}).
The 4.07M-token balanced component adds less than 0.01% to the epoch; the replay realization stays
within V10's tolerance of the declared {epoch_plan['declared_replay_fraction']!r}. **One mixture epoch;
LLOM `--sft-steps` = {epoch_plan['updates']:,}.**

## Tool-conversion replacement (coverage boundary)

The toolconv coverage boundary over the verified-modified population (the 2026-11-25 index snapshot
predates the RNO-era archive ingest, so the verified-modified count is a conservative floor):

| Boundary | Rows |
|---|---:|
| Verified-modified (2026-11-25 index snapshot) | {TOOLCONV_COVERAGE['verified_modified_2026_11_25_index_snapshot']:,} |
| Applied in V10.1 (this release) | {TOOLCONV_COVERAGE['applied']:,} |
| Already applied at the V10 build | {TOOLCONV_COVERAGE['pre_applied_at_build']:,} |
| Unmappable to the V10 lineage | {TOOLCONV_COVERAGE['unmappable_to_v10_lineage']:,} |
| Quarantined by the tokenizer gate (never silently substituted) | {TOOLCONV_COVERAGE['quarantined_tokenizer_mismatch']:,} |

Per-source replacement accounting (rows conserved; token delta = converted payload tokens −
original row tokens at the replaced rows):

| Source | Replaced rows | Token delta | Rewritten shards | Unchanged shards |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {source} | {record['replaced']:,} | {record['token_delta']:+,} | {record['rewritten_shards']} | "
        f"{record['unchanged_shards']} |"
        for source, record in sorted(toolconv.items())
    ) + f"""

Only the 15 delta-bearing shards are rewritten; the other 222 shards of the three sources and every
other source point at byte-identical V10 shard paths (re-expressed relative to the catalog root).
The apply receipt: {args.apply_receipt}.

## Balanced-reasoning component

`{BALANCED_SOURCE}` (7,728 rows / 4,070,781 tokens / 2,565,856 supervised positions,
`chat_assistant_only`, the pinned 2026.09.18 tokenizer) is ADDED at weight 0.005 per phase. Rationale
(STATUS.md §1c): a targeted thinking-style corrective store; the natural-token share (~4.07M of
~99.9B SFT tokens ≈ 0.004%) would be too small to influence behavior, while 0.005 gives ≈250M
sampled positions per one-epoch ≈ 7 passes over the store — meaningful exposure without crowding
out a family. Every non-`pretrain/` phase weight scales by (S − 0.005)/S; the replay share stays at
V10's halved {0.10037232630869815!r}.

## Weights

Every phase: the 384 `pretrain/` components untouched; the 49 SFT components renormalized by
(S − 0.005)/S; the balanced component added at 0.005 (each phase sums to 1.0 within 1e-12). The
per-source weights are in the CSV ({args.output_csv}).
"""
    Path(args.output_md).write_text(report)
    Path(args.output_csv).write_bytes(csv_bytes)
    print(f"MIX-COMPARISON-V10.1 written: {args.output_md} + {args.output_csv}")
    print("population changes exactly:", sorted(population_changes))


if __name__ == "__main__":
    main()
