# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-backed figures for the accessible Hessian report. No experiment reruns."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
if not (SOURCE / "scientific-results.json").exists():
    SOURCE = ROOT / "data"
DATA = json.loads((SOURCE / "scientific-results.json").read_text())
ROWS = list(csv.DictReader((SOURCE / "optimizer-step-results.csv").open()))
for row in ROWS:
    for k in row:
        if k != "version":
            row[k] = float(row[k])
assert len(ROWS) == 60
B = "#1F77B4"
G = "#5F6368"
O = "#FF7F0E"
INK = "#17212B"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
    }
)
FACTS = {}


def save(fig, name, values):
    fig.savefig(ROOT / (name + ".pdf"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(ROOT / (name + ".svg"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(ROOT / (name + ".png"), bbox_inches="tight", pad_inches=0.12, dpi=220)
    svg = ROOT / (name + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    FACTS[name] = values


def axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#b0b0b0", linestyle="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color("black")


def block(traj, block, d=12):
    return DATA["hessian-" + traj]["blocks"][block][str(d)]


# A concrete map separates the measurements from the motivating training comparison.
fig, ax = plt.subplots(figsize=(8.4, 4.1))
ax.set(xlim=(0, 10), ylim=(0, 4.8))
ax.axis("off")
boxes = [
    (0.08, 3.2, 2.95, 1.22, "Earlier training\ncomparison", "1/8 had a small\nfinal loss advantage"),
    (3.52, 3.2, 2.95, 1.22, "Original\nmeasurements", "3 saved states\nfrom the 1/4 run"),
    (6.96, 3.2, 2.95, 1.22, "Remaining\nquestion", "Why did 1/8\nperform better?"),
]
for x, y, w, h, t, z in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04", fill=False, edgecolor=G, lw=1.2))
    ax.text(x + w / 2, y + 0.91, t, ha="center", va="center", weight="bold", fontsize=11, linespacing=1.2)
    ax.text(x + w / 2, y + 0.33, z, ha="center", va="center", fontsize=10, linespacing=1.2)
for a, b in [(3.08, 3.46), (6.52, 6.90)]:
    ax.add_patch(FancyArrowPatch((a, 3.8), (b, 3.8), arrowstyle="-|>", mutation_scale=13, color=G))
ax.text(5, 2.67, "7 added experiments address the gaps", ha="center", fontsize=13, weight="bold")
for x, t, z in [
    (1.55, "3 motivating\ncheckpoints", "1/8 run: early,\nmiddle, late"),
    (5, "2 wider\nmeasurements", "12 directions retained\nwithin a set of 48"),
    (8.45, "2 optimizer\ncontrols", "Fresh histories\nand local loss tests"),
]:
    ax.add_patch(
        FancyBboxPatch((x - 1.46, 0.8), 2.92, 1.37, boxstyle="round,pad=.04", facecolor="#f1f6fa", edgecolor=B, lw=1.2)
    )
    ax.text(x, 1.83, t, ha="center", va="center", fontsize=11, weight="bold", linespacing=1.2)
    ax.text(x, 1.17, z, ha="center", va="center", fontsize=10, linespacing=1.2)
ax.text(5, 0.23, "All 7 completed at fixed checkpoints. No new training runs.", ha="center", fontsize=11)
save(
    fig,
    "experiment-map",
    {"original_checkpoints": 3, "added_checkpoint_measurements": 3, "added_wide_measurements": 2, "added_controls": 2},
)

# Acceptance map; explicit words make it usable without color.
configs = [
    ("1/8 early\n12 directions", "eighth-early", 12),
    ("1/8 middle\n12 directions", "eighth-middle", 12),
    ("1/8 late\n12 directions", "eighth-late", 12),
    ("1/8 middle\n48 directions", "eighth-wide", 48),
    ("1/4 middle\n12 directions", "quarter-wide", 12),
    ("1/4 middle\n48 directions", "quarter-wide", 48),
]
fig, ax = plt.subplots(figsize=(8, 4.65))
ax.set(xlim=(-2.4, 4), ylim=(-0.6, 6.1))
ax.axis("off")
records = []
for x, t in [(0.75, "Attention"), (2.8, "MLP")]:
    ax.text(x, 5.9, t, ha="center", weight="bold")
for i, (label, key, d) in enumerate(configs):
    y = 5.3 - i
    ax.text(-2.35, y, label, va="center", fontsize=12)
    for x, b in [(0.75, "attention"), (2.8, "mlp")]:
        v = block(key, b, d)
        a = v["accepted"]["gram_raw"]
        status = v["resolution"]["status"]
        if a:
            text = "Accepted"
            color = B
            bg = "#edf4f9"
            symbol = "o"
        elif key == "quarter-wide" and b == "attention":
            text = "Negative +\nmissing check"
            color = G
            bg = "#f1f1f1"
            symbol = "x"
        elif status == "indefinite":
            text = "Negative"
            color = G
            bg = "#f1f1f1"
            symbol = "x"
        else:
            text = "Unresolved"
            color = O
            bg = "#fff4e8"
            symbol = "^"
        ax.add_patch(
            FancyBboxPatch((x - 0.92, y - 0.37), 1.84, 0.74, boxstyle="round,pad=.03", facecolor=bg, edgecolor="#c7ccd1")
        )
        ax.scatter([x - 0.66], [y], marker=symbol, s=58, color=color, linewidth=1.8)
        ax.text(x + 0.12, y, text, ha="center", va="center", fontsize=11, color=INK)
        records.append({"trajectory_stage": key, "block": b, "directions": d, "accepted": a, "resolution": status})
save(fig, "acceptance-map", records)

# Accepted raw-proxy powers versus the two proposed powers.
fitrows = [
    ("1/8 early · attention", "eighth-early", "attention", 12),
    ("1/8 middle · attention", "eighth-middle", "attention", 12),
    ("1/8 middle · MLP", "eighth-middle", "mlp", 12),
    ("1/4 middle · MLP, 12", "quarter-wide", "mlp", 12),
    ("1/4 middle · MLP, 48", "quarter-wide", "mlp", 48),
]
fig, ax = plt.subplots(figsize=(8.4, 3.45))
axes(ax)
ys = np.arange(5)[::-1]
values = []
for y, (label, k, b, d) in zip(ys, fitrows, strict=True):
    v = block(k, b, d)
    assert v["accepted"]["gram_raw"]
    f = v["fits"]["gram_raw"]
    r = f["r"]
    ax.plot(r, y, "o", color=B, ms=8)
    ax.text(r + 0.07, y, f"{r:.3f}", va="center", fontsize=11)
    values.append({"label": label, "r": r, "rho": f["explained_fraction_rho"], "alpha": f["alpha_log_nonnegative"]})
for x, title in [(2, "Squared relation\nr = 2"), (4, "Fourth power\nr = 4")]:
    ax.axvline(x, color=O, lw=1.6, ls="--", ymax=0.78)
    ax.text(x, 4.65, title, ha="center", fontsize=10, color=INK)
ax.set_yticks(ys, [r[0] for r in fitrows])
ax.set(xlim=(0.8, 4.4), ylim=(-0.6, 5.55), xlabel="Fitted power r (raw proxy; accepted means only)")
ax.set_xticks([1, 2, 3, 4])
fig.subplots_adjust(left=0.32, right=0.98, bottom=0.2, top=0.98)
save(fig, "fitted-powers", values)

# Same-input leakage; numerical columns avoid exaggerating the tiny decrease.
fig, ax = plt.subplots(figsize=(8.5, 3.25))
axes(ax)
leaks = []
for y, (t, b) in zip(
    [3, 2, 1, 0], [("quarter", "attention"), ("quarter", "mlp"), ("eighth", "attention"), ("eighth", "mlp")], strict=True
):
    z = block(t + "-wide", b, 48)["nested_leakage"]
    a = z["original"]
    v = z["expanded"]
    ax.hlines(y, 0, a, color="#c8cbd0", lw=2)
    ax.plot(a, y, "o", mfc="white", mec=G, ms=10, mew=1.5)
    ax.plot(v, y, "|", color=B, ms=15, mew=2)
    leaks.append({"trajectory": t, "block": b, "12_outputs": a, "48_outputs": v})
ax.set_yticks([3, 2, 1, 0], ["1/4 · attention", "1/4 · MLP", "1/8 · attention", "1/8 · MLP"])
ax.set(xlim=(0, 1), ylim=(-0.55, 3.6), xlabel="Leakage: norm outside / norm of full response")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.legend(
    handles=[
        Line2D([], [], marker="o", mfc="white", mec=G, ls="", label="12 outputs"),
        Line2D([], [], marker="|", color=B, ms=12, ls="", label="48 outputs"),
    ],
    loc="lower right",
    frameon=True,
    fontsize=10,
)
fig.subplots_adjust(left=0.24, right=0.98, bottom=0.23, top=0.98)
save(fig, "leakage-expansion", leaks)

# Fresh-versus-saved proxy distances; each mark is an observed distance, no CI.
fig, ax = plt.subplots(figsize=(8.5, 3.55))
axes(ax)
distances = []
for y, (t, b) in zip(
    [3, 2, 1, 0], [("quarter", "attention"), ("quarter", "mlp"), ("eighth", "attention"), ("eighth", "mlp")], strict=True
):
    z = block(t + "-wide", b, 48)["proxy_distances"]["gram_raw"]
    a = z["fresh_pair_log_rms"]
    p, q = z["saved_to_fresh_log_rms"]
    assert a < min(p, q)
    ax.plot(a, y, "o", color=B, ms=8)
    ax.plot(p, y + 0.09, "s", mfc="white", mec=G, ms=7, mew=1.4)
    ax.plot(q, y - 0.09, "D", mfc="white", mec=G, ms=6, mew=1.4)
    distances.append({"trajectory": t, "block": b, **z})
ax.set_yticks([3, 2, 1, 0], ["1/4 · attention", "1/4 · MLP", "1/8 · attention", "1/8 · MLP"])
ax.set(xlim=(0, 2), ylim=(-0.7, 3.55), xlabel="Distance between centered log proxies (smaller = closer)")
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.legend(
    handles=[
        Line2D([], [], marker="o", color=B, ls="", label="Fresh 1 vs fresh 2"),
        Line2D([], [], marker="s", mfc="white", mec=G, ls="", label="Saved vs fresh 1"),
        Line2D([], [], marker="D", mfc="white", mec=G, ls="", label="Saved vs fresh 2"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.47, 1.27),
    ncol=3,
    frameon=False,
    fontsize=10,
)
fig.subplots_adjust(left=0.24, right=0.98, bottom=0.23, top=0.81)
save(fig, "history-distances", distances)

# Conditional paired loss intervals, no connected trends across candidate alpha.
labels = ["0", "1/8", "1/4", "3/8", "1/2"]
alphas = [0, 0.125, 0.25, 0.375, 0.5]
for start, label in [(0.125, "eighth"), (0.25, "quarter")]:
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
    vals = []
    chosen = [r for r in ROWS if r["starting_training_alpha"] == start and r["step_fraction"] == 1]
    lo = min(r["loss_change_lower"] for r in chosen) * 1000
    hi = max(r["loss_change_upper"] for r in chosen) * 1000
    pad = (hi - lo) * 0.1
    for ax, version, title in zip(
        axs,
        ["native", "global_norm_matched_to_alpha0"],
        ["Original update size", "Equal update norms"],
        strict=True,
    ):
        axes(ax)
        ax.axvline(0, color="black", lw=1.2)
        for y, a in zip([4, 3, 2, 1, 0], alphas, strict=True):
            r = next(r for r in chosen if r["version"] == version and r["candidate_alpha"] == a)
            m = r["loss_change"] * 1000
            l = r["loss_change_lower"] * 1000
            h = r["loss_change_upper"] * 1000
            ax.errorbar(m, y, xerr=[[m - l], [h - m]], fmt="o", color=B, ecolor=G, ms=6, capsize=3, lw=1.5)
            vals.append(r)
        ax.set_title(title, pad=11)
        ax.set(xlim=(lo - pad, hi + pad), ylim=(-0.6, 4.6))
        ax.set_yticks([4, 3, 2, 1, 0], labels)
        ax.text(0.02, -0.26, "← Better loss", transform=ax.transAxes, fontsize=10)
        ax.text(0.98, -0.26, "Worse loss →", transform=ax.transAxes, ha="right", fontsize=10)
    axs[0].set_ylabel("Candidate alpha")
    fig.supxlabel("Loss after step \u2212 unchanged loss, multiplied by 1,000", y=0.01, fontsize=12)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.29, top=0.84, wspace=0.12)
    save(fig, "optimizer-" + label, vals)

# Direct paired contrast prevents confusing overlapping marginal intervals with a paired test.
fig, ax = plt.subplots(figsize=(8.4, 3.2))
axes(ax)
ax.axvline(0, color="black", lw=1.2)
contrast = []
clabel = []
for y, (start, version) in zip(
    [3, 2, 1, 0],
    [
        (0.125, "native"),
        (0.125, "global_norm_matched_to_alpha0"),
        (0.25, "native"),
        (0.25, "global_norm_matched_to_alpha0"),
    ],
    strict=True,
):
    r = next(
        r
        for r in ROWS
        if r["starting_training_alpha"] == start
        and r["candidate_alpha"] == 0.25
        and r["version"] == version
        and r["step_fraction"] == 1
    )
    m = r["versus_candidate_eighth"] * 1000
    l = r["versus_eighth_lower"] * 1000
    h = r["versus_eighth_upper"] * 1000
    ax.errorbar(m, y, xerr=[[m - l], [h - m]], fmt="o", color=B, ecolor=G, ms=7, capsize=4)
    clabel.append(
        ("1/8 state" if start == 0.125 else "1/4 state")
        + " · "
        + ("original size" if version == "native" else "equal norms")
    )
    contrast.append(r)
ax.set_yticks([3, 2, 1, 0], clabel)
ax.set(xlim=(-0.95, 0.25), ylim=(-0.6, 3.5), xlabel="Loss with 1/4 \u2212 loss with 1/8, multiplied by 1,000")
ax.text(0.02, 1.03, "Negative favors candidate 1/4", transform=ax.transAxes, fontsize=11)
ax.text(0.98, 1.03, "Positive favors 1/8", transform=ax.transAxes, fontsize=11, ha="right")
fig.subplots_adjust(left=0.34, right=0.98, bottom=0.23, top=0.86)
save(fig, "paired-contrast", contrast)

FACTS["_sources"] = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [SOURCE / "scientific-results.json", SOURCE / "optimizer-step-results.csv"]
}
(ROOT / "figure-data.json").write_text(json.dumps(FACTS, indent=2) + "\n")
print("Exported", len(FACTS) - 1, "new figures as PDF, SVG and PNG")
