"""Compare two checkpoints on a Harbor benchmark run with several attempts per task.

Each task's score is its mean reward over scored attempts. The difference between
checkpoints is the mean over tasks scored in both, with a 95% paired bootstrap
interval over tasks (10,000 resamples, seed 0).

usage: paired_repeats.py <results root A> <results root B> [<more roots for B>...]

Extra roots are pooled into B's attempts, for example to add a single-attempt run.
"""

import random
import sys
from collections import defaultdict

from finestore.reader import ReadView
from rigging.filesystem.s3_compat import configure_coreweave_s3

RESAMPLES = 10_000


def attempts(roots: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for root in roots:
        reader = ReadView(root)
        table = next(name for name in reader.table_names() if "sample" in name)
        for row in reader.iter_rows(table):
            grading = row.get("grading") or {}
            score = grading.get("score") if isinstance(grading, dict) else None
            if score is not None:
                out[str(row["doc_id"])].append(float(score))
    return out


def main() -> None:
    configure_coreweave_s3()
    a = attempts([sys.argv[1]])
    b = attempts(sys.argv[2:])
    shared = sorted(set(a) & set(b))
    mean_a = {k: sum(a[k]) / len(a[k]) for k in shared}
    mean_b = {k: sum(b[k]) / len(b[k]) for k in shared}
    diffs = [mean_b[k] - mean_a[k] for k in shared]
    point = sum(diffs) / len(diffs)
    rng = random.Random(0)
    boot = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(RESAMPLES))
    low, high = boot[int(0.025 * RESAMPLES)], boot[int(0.975 * RESAMPLES)]
    print(f"tasks={len(shared)} attempts_a={sum(len(a[k]) for k in shared)} attempts_b={sum(len(b[k]) for k in shared)}")
    print(f"a={sum(mean_a.values()) / len(shared):.3f} b={sum(mean_b.values()) / len(shared):.3f}")
    print(f"b_minus_a={point:+.3f} ({low:+.3f} to {high:+.3f})")
    print(f"tasks_b_higher={sum(d > 0 for d in diffs)} tasks_a_higher={sum(d < 0 for d in diffs)}")


if __name__ == "__main__":
    main()
