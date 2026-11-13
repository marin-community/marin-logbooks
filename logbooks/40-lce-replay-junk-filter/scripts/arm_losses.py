"""Compare the two arms' logged losses from W&B at matching steps.

usage: arm_losses.py [pattern ...]   (substrings a metric name must contain; default: eval and train loss)
"""

import sys

import wandb

PROJECT = "marin-community/lce40-replay-filter"
RUNS = {"control": "lce40-arm-control-20261113", "filtered": "lce40-arm-filtered-20261113"}


def main() -> None:
    patterns = sys.argv[1:] or ["loss"]
    api = wandb.Api()
    histories = {}
    for arm, run_id in RUNS.items():
        run = api.run(f"{PROJECT}/{run_id}")
        keys = sorted(k for k in run.summary.keys() if all(p in k for p in patterns) and not k.startswith("_"))
        print(f"{arm}: state {run.state}, last step {run.summary.get('_step')}, {len(keys)} matching metrics")
        histories[arm] = (run, keys)
    shared = [k for k in histories["control"][1] if k in set(histories["filtered"][1])]
    print("metrics:", shared[:60])
    rows = {}
    for arm, (run, _) in histories.items():
        for key in shared[:40]:
            for record in run.scan_history(keys=["_step", key]):
                rows.setdefault(key, {}).setdefault(record["_step"], {})[arm] = record[key]
    for key in shared[:40]:
        steps = sorted(s for s, v in rows.get(key, {}).items() if len(v) == 2)
        if not steps:
            continue
        picks = steps if len(steps) <= 6 else [steps[i] for i in sorted({0, len(steps) // 4, len(steps) // 2, 3 * len(steps) // 4, len(steps) - 1})]
        cells = "  ".join(
            f"{s}: {rows[key][s]['control']:.4f}/{rows[key][s]['filtered']:.4f}" for s in picks
        )
        print(f"{key}\n    step: control/filtered  {cells}")


if __name__ == "__main__":
    main()
