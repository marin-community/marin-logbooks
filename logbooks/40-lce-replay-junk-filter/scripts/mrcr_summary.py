"""Print the headline needle-recovery numbers of finished Snowball MRCR runs.

usage: mrcr_summary.py <run id> [<run id> ...]   (run ids as given to launch_snowball_mrcr_eval)
"""

import json
import sys

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

ROOT = "s3://marin-us-east-08a/marin/eval/mrcr-snowball"
PREFIX = "eval/mrcr/two_shot/cap_262144/"
WANTED = (
    "examples",
    "micro_needle_recovery_ratio",
    "micro_needle_recovery_ratio_ci95_low",
    "micro_needle_recovery_ratio_ci95_high",
    "micro_isolated_needle_gain_nll",
    "micro_distracted_needle_gain_nll",
    "micro_context_gain_nll",
    "full_context/micro_loss",
    "query_only/micro_loss",
)


def main() -> None:
    configure_coreweave_s3()
    for run in sys.argv[1:]:
        path = StoragePath.parse(f"{ROOT}/{run}/2026.11.13/mrcr_metrics.json")
        if not path.exists():
            print(f"{run}: no results yet")
            continue
        record = json.loads(path.read_text())
        metrics = record["metrics"]
        print(f"\n{run}  (checkpoint step {record['checkpoint_step']}, max_batches {record['max_batches']})")
        for name in WANTED:
            if PREFIX + name in metrics:
                print(f"  {name:42s} {metrics[PREFIX + name]:.5f}")
        if PREFIX + WANTED[1] not in metrics:
            top = sorted(k[len(PREFIX):] for k in metrics if k.startswith(PREFIX) and k.count("/") == PREFIX.count("/"))
            print("  top-level metric names:", top[:40])


if __name__ == "__main__":
    main()
