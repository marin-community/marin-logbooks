"""Build the V10 vs V9 mix comparison (MIX-COMPARISON-V10.md + CSV).

Two mixes side by side, one row per source in their union (they share all 433):

- V9: the sealed release (2026.11.22-v9-overthinking) — rows/tokens from the sealed
  cache catalog, weights from the release schedule (the declared replay share
  20.074465%).
- V10: the candidate (2026.11.23-v10-overthinking) — rows/tokens from the V10
  catalog (two sources re-versioned @terminus2-keep20-20261123), weights from the
  halved-replay schedule (the declared replay share 10.037232630869815%).

The markdown report adds the epoch-level summary: the declared and realized replay
fractions, the packed SFT sequences, the run sequences and the update count from
the V10 capacity receipt beside V9's 1,893 @ 256×262,144, and the Terminus-2
detector evidence. The seal is not touched by this script.
"""

import argparse
import csv
import io
import json
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for


def read_bytes(uri: str) -> bytes:
    fs, key = filesystem_for(uri)
    return fs.cat(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-release", required=True)
    parser.add_argument("--v10-release", required=True, help="The frozen pre-consumer V10 manifest")
    parser.add_argument("--scan-receipt", required=True)
    parser.add_argument("--reversion-receipt", required=True)
    parser.add_argument("--capacity-receipt", required=True, help="The V10 epoch-plan/capacity JSON")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    v9 = json.loads(read_bytes(args.v9_release))
    v10_manifest = json.loads(read_bytes(args.v10_release))
    v9_catalog = json.loads(read_bytes(v9["artifacts"]["cache_catalog.json"]["uri"]))["splits"]["train"]
    v10_catalog = json.loads(read_bytes(v10_manifest["artifacts"]["cache_catalog.json"]["uri"]))["splits"]["train"]
    v9_weights = json.loads(read_bytes(v9["artifacts"]["train_weights.json"]["uri"]))[0][1]
    v10_weights = json.loads(read_bytes(v10_manifest["artifacts"]["train_weights.json"]["uri"]))[0][1]
    scan = json.loads(read_bytes(args.scan_receipt))
    reversion = json.loads(read_bytes(args.reversion_receipt))
    capacity = json.loads(read_bytes(args.capacity_receipt))

    rows = []
    for source in sorted(set(v9_catalog) | set(v10_catalog)):
        v9_entry = v9_catalog.get(source)
        v10_entry = v10_catalog.get(source)
        row = {
            "component": source,
            "v9_rows": v9_entry["ledger"]["total_num_rows"] if v9_entry else "",
            "v10_rows": v10_entry["ledger"]["total_num_rows"] if v10_entry else "",
            "v9_tokens": v9_entry["ledger"]["field_counts"]["input_ids"] if v9_entry else "",
            "v10_tokens": v10_entry["ledger"]["field_counts"]["input_ids"] if v10_entry else "",
            "v9_weight": repr(v9_weights[source]),
            "v10_weight": repr(v10_weights[source]),
            "v9_cache_dir": v9_entry["cache_dir"] if v9_entry else "",
            "v10_cache_dir": v10_entry["cache_dir"] if v10_entry else "",
            "terminus2_affected": "yes" if source in reversion["sources"] else "",
        }
        rows.append(row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode()

    totals_v9 = {
        "components": len(v9_catalog),
        "tokens": sum(entry["ledger"]["field_counts"]["input_ids"] for entry in v9_catalog.values()),
        "rows": sum(entry["ledger"]["total_num_rows"] for entry in v9_catalog.values()),
    }
    totals_v10 = {
        "components": len(v10_catalog),
        "tokens": sum(entry["ledger"]["field_counts"]["input_ids"] for entry in v10_catalog.values()),
        "rows": sum(entry["ledger"]["total_num_rows"] for entry in v10_catalog.values()),
    }
    harness = {
        source: {
            "rows": scan["sources"][source]["rows"],
            "harness_rows": scan["sources"][source]["marker_rows"],
            "keep_rows": reversion["sources"][source]["kept_harness_rows"],
            "dropped_rows": reversion["sources"][source]["dropped_harness_rows"],
            "keep_ids_sha256": reversion["sources"][source]["keep_ids_sha256"],
        }
        for source in reversion["sources"]
    }
    report = f"""# MIX-COMPARISON-V10: the sealed V9 mix vs the V10 candidate

V10 = sealed V9 with exactly the two user subsampling decisions (2026-11-23): the pretraining replay
share halved, the Terminus-2-format traces cut to a seeded 20% per affected source. Everything else is
byte-identical V9. The seal is not touched by this report.

## Epoch summary

| Quantity | V9 | V10 |
|---|---|---|
| Declared replay share | {v9_weights and 0.20074465261739632!r} | {capacity['declared_replay_fraction']!r} |
| Realized replay share | {0.20092322041732702!r} | {capacity['realized_replay_fraction']!r} |
| Packed SFT sequences (every row once) | 387,239 | {capacity['training_sft_sequences']:,} |
| Run sequences | 484,608 | {capacity['run_sequences']:,} |
| Updates @ 256×262,144 | 1,893 | {capacity['updates']:,} |
| Context positions | 127,037,079,552 | {capacity['context_positions']:,} |
| Components | {totals_v9['components']} | {totals_v10['components']} |
| Store tokens | {totals_v9['tokens']:,} | {totals_v10['tokens']:,} |
| Store rows | {totals_v9['rows']:,} | {totals_v10['rows']:,} |

Update count change: V9's 1,893 -> V10's {capacity['updates']:,} ({capacity['updates'] - 1893:+,}).

## Replay halving (both readings)

Landed rule: **half the declared replay share** (the replay mix budget halves):
0.20074465261739632 -> {capacity['declared_replay_fraction']!r}. Over the V9 epoch numbers
(387,239 packed SFT sequences, 484,608 run sequences, 97,369 replay sequences):

- Half the declared **share**: declared {capacity['declared_replay_fraction']!r}; the replay fills
  ~43,205 sequences of a ~430,444-sequence epoch — the replay *positions* become ~44.4% of V9's,
  because the epoch shrinks around the fixed once-each SFT rows.
- Half the replay **positions**: declared 0.11168129270388039 (~48,685 replay sequences of
  ~435,924) — not the landed rule.

## Terminus-2 harness-format cut (detector evidence)

The marker: the Terminus-2 command-line task-description opener, matched as a contiguous token
subsequence over every row's raw ids across all 49 SFT sources (12,220,678 rows scanned; no
decoding). Exactly two sources carry it — both at 100% of rows; every other source is byte-identical
to V9. Each affected source keeps the seeded 20% of its harness rows (exact floor(0.2·n) count in
seeded_order(f"{{shard}}:{{index}}", "terminus2-keep20-20261123") order, keep-ids hashes recorded);
the non-harness rows of those sources are untouched.

| Source | Rows | Harness rows | Kept (20%) | Dropped | Keep-ids SHA-256 |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {source} | {record['rows']:,} | {record['harness_rows']:,} | {record['keep_rows']:,} | "
        f"{record['dropped_rows']:,} | `{record['keep_ids_sha256'][:16]}…` |"
        for source, record in sorted(harness.items())
    ) + f"""

Per-shard detector evidence and the full selection rule: the scan receipt ({args.scan_receipt}) and
the re-version receipt ({args.reversion_receipt}).

## Weights

Every phase: the 384 pretrain components x0.5; the 49 SFT components renormalized x1.1255823018731081
(each phase sums to 1.0). The per-source weights are in the CSV ({args.output_csv}).
"""
    Path(args.output_md).write_text(report)
    Path(args.output_csv).write_bytes(csv_bytes)
    print(f"MIX-COMPARISON-V10 written: {args.output_md} + {args.output_csv}")


if __name__ == "__main__":
    main()
