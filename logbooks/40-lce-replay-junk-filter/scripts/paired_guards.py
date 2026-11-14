"""Dump per-task rewards from Harbor eval archives and pair two arms.

usage: guard_samples.py <control results root> <filtered results root>
"""

import math
import sys

from finestore.reader import ReadView
from rigging.filesystem.s3_compat import configure_coreweave_s3

configure_coreweave_s3()


def rewards(root: str) -> dict[str, float]:
    reader = ReadView(root)
    names = reader.table_names()
    table = next(name for name in names if "sample" in name)
    out: dict[str, float] = {}
    first = True
    for row in reader.iter_rows(table):
        if first:
            print("tables", names, "columns", sorted(row), file=sys.stderr)
            print({k: str(v)[:80] for k, v in row.items()}, file=sys.stderr)
            first = False
        grading = row.get("grading") or {}
        score = grading.get("score") if isinstance(grading, dict) else None
        if score is None and row.get("correct") is not None:
            score = float(row["correct"])
        out[str(row["doc_id"])] = score
    return out


control, filtered = rewards(sys.argv[1]), rewards(sys.argv[2])
shared = sorted(k for k in control if k in filtered and control[k] is not None and filtered[k] is not None)
both = sum(control[k] > 0 and filtered[k] > 0 for k in shared)
only_control = sum(control[k] > 0 and not filtered[k] > 0 for k in shared)
only_filtered = sum(filtered[k] > 0 and not control[k] > 0 for k in shared)
neither = len(shared) - both - only_control - only_filtered
n = only_control + only_filtered
# Exact two-sided sign (McNemar) test on discordant pairs.
tail = sum(math.comb(n, i) for i in range(0, min(only_control, only_filtered) + 1)) / 2**n if n else 1.0
print(f"control={len(control)} filtered={len(filtered)} paired={len(shared)}")
print(f"both={both} only_control={only_control} only_filtered={only_filtered} neither={neither}")
print(f"control_rate={(both + only_control) / len(shared):.3f} filtered_rate={(both + only_filtered) / len(shared):.3f}")
print(f"mcnemar_exact_p={min(1.0, 2 * tail):.3f}")
