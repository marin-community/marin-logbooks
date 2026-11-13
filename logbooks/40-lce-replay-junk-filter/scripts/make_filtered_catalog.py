"""Write the filtered arm's cache catalog: the production catalog with rebuilt long replay caches swapped in.

A training entry whose cache is a cell's control long cache is pointed at the filtered cache and given
that cache's own ledger. Validation entries and every other training entry are unchanged. Also writes a
summary of what the swap did to the long replay bucket.

usage: make_filtered_catalog.py <production catalog> <filtered replay prefix> <local summary.jsonl out>
"""

import json
import re
import sys

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

REPLAY_LONG = re.compile(
    r"s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/replay/(c\d\dq\d-train)/2026\.10\.24\.1/gt_64k"
)


def main() -> None:
    configure_coreweave_s3()
    production, prefix, summary_out = sys.argv[1:4]
    root = StoragePath.parse(prefix)
    summaries = [json.loads(line) for line in (root / "summary.jsonl").read_text().splitlines()]
    open(summary_out, "w").write("".join(json.dumps(row) + "\n" for row in summaries))
    built = {row["cell"]: row for row in summaries if row["kept_documents"] > 0}
    catalog = json.loads(StoragePath.parse(production).read_text())
    ledgers: dict[str, dict] = {}
    swapped = 0
    control_tokens = filtered_tokens = 0
    for entry in catalog["splits"]["train"].values():
        match = REPLAY_LONG.fullmatch(entry["cache_dir"])
        if match is None or match.group(1) not in built:
            continue
        cell = match.group(1)
        new_dir = f"{prefix}/{cell}/gt_64k"
        if cell not in ledgers:
            ledgers[cell] = json.loads((root / cell / "gt_64k" / "shard_ledger.json").read_text())
            control_tokens += entry["ledger"]["field_counts"]["input_ids"]
            filtered_tokens += ledgers[cell]["field_counts"]["input_ids"]
        entry["cache_dir"] = new_dir
        entry["ledger"] = ledgers[cell]
        swapped += 1
    (root / "cache_catalog.json").write_text(json.dumps(catalog))
    judged = sum(row["judged_documents"] for row in summaries)
    flagged = sum(row["flagged_documents"] for row in summaries)
    short = [row["cell"] for row in summaries if row["kept_tokens"] < row["budget_tokens"]]
    changed = [row["cell"] for row in summaries if row["flagged_documents"] > 0]
    print(f"cells built {len(built)} of {len(summaries)}; catalog entries swapped {swapped}")
    print(f"long tokens: control {control_tokens/1e9:.3f}B, filtered {filtered_tokens/1e9:.3f}B")
    print(f"documents judged {judged}, flagged {flagged} ({flagged/max(1, judged):.1%})")
    print(f"flagged tokens skipped {sum(row['flagged_tokens'] for row in summaries)/1e9:.3f}B")
    print(f"cells whose selection changed {len(changed)}; cells short of budget {len(short)}: {short[:12]}")


if __name__ == "__main__":
    main()
