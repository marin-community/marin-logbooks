"""Measure how much of the control long replay the filter replaces.

The builder judges long documents in the control's own order, so the control selection of a cell is the
prefix of its verdicts that first crosses the cell's budget. This reports the flagged share of that prefix.

usage: treatment_dose.py <filtered replay prefix> <cells yaml.gz>
"""

import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.sft_cold_start.filtered_replay import read_cells


def main() -> None:
    configure_coreweave_s3()
    root = StoragePath.parse(sys.argv[1])
    cells = read_cells(sys.argv[2])

    def control_prefix(cell):
        rows = [json.loads(line) for line in (root / cell.name / "verdicts.jsonl").read_text().splitlines()]
        prefix, total = [], 0
        for row in rows:
            if total >= cell.long_tokens:
                break
            prefix.append(row)
            total += row["tokens"]
        return cell.name, prefix

    with ThreadPoolExecutor(max_workers=16) as pool:
        prefixes = dict(pool.map(control_prefix, cells))
    docs = tokens = flagged_docs = flagged_tokens = 0
    by_category = collections.Counter()
    by_quality = collections.defaultdict(lambda: [0, 0])
    for name, prefix in prefixes.items():
        for row in prefix:
            docs += 1
            tokens += row["tokens"]
            by_quality[name[3:5]][1] += row["tokens"]
            if row["category"] is not None:
                flagged_docs += 1
                flagged_tokens += row["tokens"]
                by_category[row["category"]] += row["tokens"]
                by_quality[name[3:5]][0] += row["tokens"]
    print(f"control long selection: {docs} documents, {tokens/1e9:.3f}B tokens across {len(prefixes)} cells")
    print(f"flagged: {flagged_docs} documents ({flagged_docs/docs:.1%}), {flagged_tokens/1e9:.3f}B tokens ({flagged_tokens/tokens:.1%})")
    for category, count in by_category.most_common():
        print(f"  {category:18s} {count/tokens:6.1%} of control long tokens")
    for quality, (flagged, total) in sorted(by_quality.items()):
        print(f"  quality {quality}: {flagged/total:6.1%} of {total/1e9:.3f}B tokens flagged")


if __name__ == "__main__":
    main()
