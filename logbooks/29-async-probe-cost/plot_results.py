# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the completed paired comparison without treating cycles as replicates."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot(analysis_path: Path, output: Path) -> None:
    analysis = json.loads(analysis_path.read_text())
    primary = analysis["comparisons"]["10"]["complete_cycle_seconds"]
    assert primary["pairs"] == 3
    runs = {(run["seed"], run["interval"]): run for run in analysis["runs"]}
    seeds = (1701, 2903, 4517)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, (times, changes) = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw={"width_ratios": [1, 1.2]})
    colors = {0: "#2864a0", 10: "#ba5c12"}
    for index, seed in enumerate(seeds):
        control = runs[seed, 0]["complete_cycle_seconds"] / 60
        probe = runs[seed, 10]["complete_cycle_seconds"] / 60
        times.plot([index - 0.1, index + 0.1], [control, probe], color="#a0a0a0", linewidth=1.2)
        times.scatter(index - 0.1, control, color=colors[0], s=45, label="Probes off" if index == 0 else None)
        times.scatter(
            index + 0.1, probe, color=colors[10], s=50, marker="^", label="Interval 10" if index == 0 else None
        )
    times.set_xticks(range(3), [str(seed) for seed in seeds])
    times.set_xlabel("Matched starting seed")
    times.set_ylabel("Complete 30-update window (minutes)")
    times.set_title("Observed elapsed time", loc="left", fontsize=12)
    times.legend(frameon=False, loc="lower left")
    times.grid(axis="y", alpha=0.2)

    pair_changes = [100 * (ratio - 1) for ratio in primary["paired_ratios"]]
    changes.scatter(pair_changes, [3, 2, 1], color="#2864a0", s=45)
    mean = primary["added_percent"]
    changes.errorbar(
        mean,
        0,
        xerr=[[mean - primary["lower95_percent"]], [primary["upper95_percent"] - mean]],
        fmt="D",
        color="#172d43",
        capsize=4,
        markersize=6,
    )
    changes.axvline(0, color="#808080", linewidth=1)
    changes.axvline(1, color="#b15425", linewidth=1, linestyle="--", label="1% target")
    changes.set_yticks([3, 2, 1, 0], [str(seed) for seed in seeds] + ["Geometric mean"])
    changes.set_xlabel("Added complete-cycle time (%)")
    changes.set_title("Three pairs; 95% t interval on log ratios", loc="left", fontsize=11)
    changes.legend(frameon=False, loc="upper left")
    changes.grid(axis="x", alpha=0.2)
    changes.set_xlim(min(-8.5, primary["lower95_percent"] - 1), max(7.5, primary["upper95_percent"] + 1))
    figure.suptitle("Periodic probes during asynchronous Grug PPO/TIS", fontsize=14, x=0.04, ha="left")
    figure.text(
        0.04,
        0.02,
        "40 H100s per window; 30 nonzero updates per arm. Response and scheduling variation remain.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.92))
    for suffix in ("png", "svg"):
        figure.savefig(output / ("paired-cycle-time." + suffix), dpi=200, bbox_inches="tight")
    svg_path = output / "paired-cycle-time.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.analysis, args.output)
