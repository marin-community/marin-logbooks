"""Build the V10.2 vs V10.1 mix comparison (MIX-COMPARISON-V10.2.md + CSV).

Two mixes side by side, one row per source in their union (both are V10's 433
plus the balanced component = 434):

- V10.1: the sealed release (2026.11.25-v10-1-balanced-toolconv/release-v10.1) —
  rows/tokens from the sealed cache catalog, weights from the sealed schedule,
  1,680 updates @ 256×262,144.
- V10.2: the candidate — rows/tokens from the V10.2 catalog (the eight toolconv
  sources re-versioned @toolconv-20261127 over stores-v4), weights byte-identical
  to V10.1's.

The markdown report adds the toolconv merge boundary (42,272 applied, the 40,374
same-name rows as the named design exclusion), the per-source accounting and the
epoch-level summary from the V10.2 capacity receipt and epoch plan beside V10.1's
numbers. The seal is not touched by this script.
"""

import argparse
import csv
import io
import json
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for

REPLACED_SOURCES = (
    "sft/source/nemotron_sft_v3/math_v3/train",
    "sft/source/nemotron_sft_v3/math_v4/train",
    "sft/source/nemotron_sft_v3/opencode_v1/agent_skills",
    "sft/source/nemotron_sft_v3/opencode_v1/bash_only_tool",
    "sft/source/nemotron_sft_v3/opencode_v1/bash_only_tool_skills",
    "sft/source/nemotron_sft_v3/opencode_v1/question_tool",
    "sft/source/nemotron_sft_v3/swe_v1/r2e_gym",
    "sft/source/nemotron_sft_v3/swe_v2/openhands_swe",
)
EXPECTED_TOTAL_REPLACED = 42272
EXPECTED_TOKEN_DELTA = -132820164
NAMED_EXCLUSION_ROWS = 40374
V10_1_UPDATES = 1680
TOOLCONV_COVERAGE = {
    "applied": 42272,
    "lineage_not_applied": 40698,
    "args_check_not_applied": 1574,
    "retokenized_from_text": 13958,
    "same_name_design_exclusion_byte_identical": 40374,
    "verified_modified_2026_11_27_index_snapshot": 2223483,
    "applied_baked_content_confirmed": 1253844,
    "tracer_conflicts": 28787,
}


def read_bytes(uri: str) -> bytes:
    fs, key = filesystem_for(uri)
    return fs.cat(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10-1-release", required=True)
    parser.add_argument("--v10-2-manifest", required=True, help="The frozen pre-consumer V10.2 manifest")
    parser.add_argument("--apply-receipt", required=True)
    parser.add_argument("--v10-1-epoch-plan", required=True, help="The sealed V10.1 epoch-plan.json")
    parser.add_argument("--capacity-receipt", required=True, help="The V10.2 capacity.json")
    parser.add_argument("--epoch-plan", required=True, help="The V10.2 epoch-plan.json")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    v10_1 = json.loads(read_bytes(args.v10_1_release))
    v10_2_manifest = json.loads(read_bytes(args.v10_2_manifest))
    v10_1_catalog = json.loads(read_bytes(v10_1["artifacts"]["cache_catalog.json"]["uri"]))["splits"]["train"]
    v10_2_catalog = json.loads(read_bytes(v10_2_manifest["artifacts"]["cache_catalog.json"]["uri"]))["splits"]["train"]
    v10_1_weights = json.loads(read_bytes(v10_1["artifacts"]["train_weights.json"]["uri"]))
    v10_2_weights = json.loads(read_bytes(v10_2_manifest["artifacts"]["train_weights.json"]["uri"]))
    apply_receipt = json.loads(read_bytes(args.apply_receipt))
    v10_1_epoch = json.loads(read_bytes(args.v10_1_epoch_plan))
    capacity = json.loads(read_bytes(args.capacity_receipt))
    epoch_plan = json.loads(read_bytes(args.epoch_plan))

    if v10_1_weights != v10_2_weights:
        raise ValueError("the V10.2 weights are not byte-identical to V10.1's")

    rows = []
    for source in sorted(set(v10_1_catalog) | set(v10_2_catalog)):
        v10_1_entry = v10_1_catalog.get(source)
        v10_2_entry = v10_2_catalog.get(source)
        row = {
            "component": source,
            "v10_1_rows": v10_1_entry["ledger"]["total_num_rows"] if v10_1_entry else "",
            "v10_2_rows": v10_2_entry["ledger"]["total_num_rows"] if v10_2_entry else "",
            "v10_1_tokens": v10_1_entry["ledger"]["field_counts"]["input_ids"] if v10_1_entry else "",
            "v10_2_tokens": v10_2_entry["ledger"]["field_counts"]["input_ids"] if v10_2_entry else "",
            "v10_1_weight": repr(v10_1_weights[0][1][source]) if source in v10_1_weights[0][1] else "",
            "v10_2_weight": repr(v10_2_weights[0][1][source]) if source in v10_2_weights[0][1] else "",
            "v10_1_cache_dir": v10_1_entry["cache_dir"] if v10_1_entry else "",
            "v10_2_cache_dir": v10_2_entry["cache_dir"] if v10_2_entry else "",
            "toolconv_replaced_rows": (
                apply_receipt["sources"][source]["replaced_rows"] if source in REPLACED_SOURCES else ""
            ),
        }
        rows.append(row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode()

    # The population check: rows conserve everywhere; exactly the eight toolconv
    # sources' tokens move; no source is added or dropped.
    row_changes = {row["component"] for row in rows if row["v10_1_rows"] != row["v10_2_rows"]}
    token_changes = {
        row["component"]
        for row in rows
        if row["v10_1_tokens"] != row["v10_2_tokens"]
    }
    if row_changes:
        raise ValueError(f"rows are not conserved for: {sorted(row_changes)}")
    if token_changes != set(REPLACED_SOURCES):
        raise ValueError(f"tokens differ outside the eight expected sources: {token_changes ^ set(REPLACED_SOURCES)}")

    totals_v10_1 = {
        "components": len(v10_1_catalog),
        "tokens": sum(entry["ledger"]["field_counts"]["input_ids"] for entry in v10_1_catalog.values()),
        "rows": sum(entry["ledger"]["total_num_rows"] for entry in v10_1_catalog.values()),
    }
    totals_v10_2 = {
        "components": len(v10_2_catalog),
        "tokens": sum(entry["ledger"]["field_counts"]["input_ids"] for entry in v10_2_catalog.values()),
        "rows": sum(entry["ledger"]["total_num_rows"] for entry in v10_2_catalog.values()),
    }
    token_move = totals_v10_1["tokens"] - totals_v10_2["tokens"]
    if token_move != -EXPECTED_TOKEN_DELTA:
        raise ValueError(f"the store token move is {token_move}, expected {-EXPECTED_TOKEN_DELTA}")

    toolconv = {
        source: {
            "replaced": apply_receipt["sources"][source]["replaced_rows"],
            "token_delta": apply_receipt["sources"][source]["output"]["token_delta"],
            "rewritten_shards": apply_receipt["sources"][source]["output"]["rewritten_shards"],
            "unchanged_shards": apply_receipt["sources"][source]["output"]["unchanged_shards"],
        }
        for source in REPLACED_SOURCES
    }
    total_replaced = sum(record["replaced"] for record in toolconv.values())
    total_delta = sum(record["token_delta"] for record in toolconv.values())
    if total_replaced != EXPECTED_TOTAL_REPLACED or total_delta != EXPECTED_TOKEN_DELTA:
        raise ValueError(f"the receipt totals disagree: {total_replaced} rows, {total_delta} tokens")
    rewritten_total = sum(record["rewritten_shards"] for record in toolconv.values())
    unchanged_total = sum(record["unchanged_shards"] for record in toolconv.values())

    report = f"""# MIX-COMPARISON-V10.2: the sealed V10.1 mix vs the V10.2 candidate

V10.2 = sealed V10.1 with exactly one change (2026-11-27): 42,272 verified tool-converted rows
replace their originals in the eight toolconv-delta sources (40,698 lineage not_applied +
1,574 args-check not_applied; 13,958 re-tokenized from text with the pinned 2026.09.18
tokenizer), with V10's assistant-only supervision preserved (the all-position loss for Exp7
comes from the LLOM consumption view, PR #12284). The 40,374 same-name-rewrite rows stay
byte-identical V10 rows as a NAMED design exclusion (coordinator ruling 2026-11-27, option b;
resolved in V10.3). Everything else is byte-identical V10.1 — same weights, same supervision
sidecar, same 434 components. The seal is not touched by this report.

## Epoch summary

| Quantity | V10.1 | V10.2 |
|---|---|---|
| Declared replay share | {v10_1_epoch['declared_replay_fraction']!r} | {epoch_plan['declared_replay_fraction']!r} |
| Realized replay share | {v10_1_epoch['realized_replay_fraction']!r} | {epoch_plan['realized_replay_fraction']!r} |
| Packed SFT sequences (every row once) | {v10_1_epoch['training_sft_sequences']:,} | {epoch_plan['training_sft_sequences']:,} |
| Run sequences | {v10_1_epoch['run_sequences']:,} | {epoch_plan['run_sequences']:,} |
| Updates @ 256×262,144 | {v10_1_epoch['updates']:,} | {epoch_plan['updates']:,} |
| Context positions | {v10_1_epoch['context_positions']:,} | {epoch_plan['context_positions']:,} |
| Reserved validation sequences | {v10_1_epoch['reserved_validation_sequences']} | {epoch_plan['reserved_validation_sequences']} |
| Components | {totals_v10_1['components']} | {totals_v10_2['components']} |
| Store tokens | {totals_v10_1['tokens']:,} | {totals_v10_2['tokens']:,} |
| Store rows | {totals_v10_1['rows']:,} | {totals_v10_2['rows']:,} |

Update count change: V10.1's {v10_1_epoch['updates']:,} -> V10.2's {epoch_plan['updates']:,}
({epoch_plan['updates'] - v10_1_epoch['updates']:+,}). The merge removes {token_move:,} serialized
tokens (the converted payloads are more compact than the raw tool blocks), so the packed SFT
sequence count moves by {epoch_plan['training_sft_sequences'] - v10_1_epoch['training_sft_sequences']:+,}.
Rows conserve exactly ({totals_v10_1['rows']:,} -> {totals_v10_2['rows']:,}). **One mixture epoch;
LLOM `--sft-steps` = {epoch_plan['updates']:,}.**

## Tool-conversion merge (coverage boundary)

The reconciliation over the verified-modified population (tracer r10 partitions all
{TOOLCONV_COVERAGE['verified_modified_2026_11_27_index_snapshot']:,} rows exactly; the retention
CSV residual is attributed by named cause — duplicate verifications + RNO-era growth):

| Boundary | Rows |
|---|---:|
| Verified-modified (2026-11-27 index snapshot) | {TOOLCONV_COVERAGE['verified_modified_2026_11_27_index_snapshot']:,} |
| Applied in V10.2 (this release) | {TOOLCONV_COVERAGE['applied']:,} |
| — lineage not_applied | {TOOLCONV_COVERAGE['lineage_not_applied']:,} |
| — args-check not_applied | {TOOLCONV_COVERAGE['args_check_not_applied']:,} |
| — of which re-tokenized from text | {TOOLCONV_COVERAGE['retokenized_from_text']:,} |
| Applied and baked at earlier generations | {TOOLCONV_COVERAGE['applied_baked_content_confirmed']:,} |
| Same-name rewrites: NAMED DESIGN EXCLUSION (byte-identical V10 rows; resolved in V10.3) | {TOOLCONV_COVERAGE['same_name_design_exclusion_byte_identical']:,} |
| Tracer conflicts (resolved to named causes) | {TOOLCONV_COVERAGE['tracer_conflicts']:,} |

Per-source merge accounting (rows conserved; token delta = converted payload tokens − original
row tokens at the replaced rows):

| Source | Replaced rows | Token delta | Rewritten shards | Unchanged shards |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {source} | {record['replaced']:,} | {record['token_delta']:+,} | {record['rewritten_shards']} | "
        f"{record['unchanged_shards']} |"
        for source, record in sorted(toolconv.items())
    ) + f"""
| **Total** | **{total_replaced:,}** | **{total_delta:+,}** | **{rewritten_total}** | **{unchanged_total}** |

Only the {rewritten_total} delta-bearing shards are rewritten (stores-v4); the other {unchanged_total}
shards of the eight sources and every other source point at byte-identical V10/V10.1 shard paths
(re-expressed relative to the catalog root). The apply receipt: {args.apply_receipt}.

## Named design exclusion

The {NAMED_EXCLUSION_ROWS:,} same-name-rewrite rows (opencode_v1 sources) stay byte-identical V10
rows in V10.2: the conversion campaign produced rewritten-arguments payloads, but the store's
before/after sha256 pins match no rendered/converted-argument hash under raw-string or
sorted-JSON canonicalizations (canonical-hash check /benfeuer/v10-2-canonical-hash-check-20261127,
3h20m, all 40,374 mismatch). Coordinator ruling 2026-11-27 (option b): named design exclusion,
recorded in the seal notes, resolved in V10.3.

## Weights

Byte-identical V10.1 weights (verified equal in this script): the 384 `pretrain/` components and
the 50 SFT components (including the balanced component at 0.005) carry V10.1's exact values; the
merge changes row contents, not the mixture. The per-source weights are in the CSV ({args.output_csv}).
"""
    Path(args.output_md).write_text(report)
    Path(args.output_csv).write_bytes(csv_bytes)
    print(f"MIX-COMPARISON-V10.2 written: {args.output_md} + {args.output_csv}")
    print("token changes exactly:", sorted(token_changes))
    print(f"store tokens: {totals_v10_1['tokens']:,} -> {totals_v10_2['tokens']:,} ({token_move:+,})")
    print(f"updates: {v10_1_epoch['updates']:,} -> {epoch_plan['updates']:,}")


if __name__ == "__main__":
    main()
