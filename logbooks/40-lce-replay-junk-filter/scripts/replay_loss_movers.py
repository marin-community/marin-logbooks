# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank held-out replay documents by how far the filtered arm moves from the two control seeds.

The gap is the filtered arm's per-token loss minus the mean of the two controls'. A document is
filtered-specific when the gap exceeds 0.05 nats per token and the seeds differ by less than half of it;
otherwise any movement is no larger than seed-to-seed variation on that document. Also prints group
losses with the encoded-blob documents and the single-checkpoint breakdowns set aside.

usage: replay_loss_movers.py [top-n]
"""

import json
import sys

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

ROOT = "s3://marin-us-east-02a/marin/eval/replay-loss-snowball"
VERSION = "2026.11.16"
RUNS = {
    "base": "lce40-replay-loss-base-s0",
    "c0": "lce40-replay-loss-control-s238",
    "c1": "lce40-replay-loss-control-seed1-s238",
    "f": "lce40-replay-loss-filtered-s238",
}
GAP_THRESHOLD = 0.05
BREAKDOWN_MARGIN = 0.5


def load(run: str) -> dict[tuple[str, int], dict]:
    path = StoragePath.parse(f"{ROOT}/{run}/{VERSION}/replay_document_losses.jsonl")
    return {(row["cell"], row["index"]): row for row in map(json.loads, path.read_text().splitlines())}


def document_rows() -> list[dict]:
    data = {name: load(run) for name, run in RUNS.items()}
    rows = []
    for key, reference in sorted(data["c0"].items()):
        tokens = reference["scored_tokens"]
        row = {"cell": key[0], "index": key[1], "category": reference["category"] or "kept", "tokens": tokens}
        for name in RUNS:
            row[f"sum_{name}"] = data[name][key]["loss_sum"]
            row[name] = data[name][key]["loss_sum"] / tokens
        row["gap"] = row["f"] - (row["c0"] + row["c1"]) / 2
        row["seed"] = abs(row["c1"] - row["c0"])
        row["specific"] = abs(row["gap"]) > GAP_THRESHOLD and row["seed"] < abs(row["gap"]) / 2
        trained = {name: row[name] for name in ("c0", "c1", "f")}
        row["breakdown"] = any(
            value - max(v for other, v in trained.items() if other != name) > BREAKDOWN_MARGIN
            for name, value in trained.items()
        )
        rows.append(row)
    return rows


def group_losses(rows: list[dict]) -> str:
    tokens = sum(r["tokens"] for r in rows)
    return " ".join(f"{name}={sum(r[f'sum_{name}'] for r in rows) / tokens:.4f}" for name in RUNS)


def main() -> None:
    configure_coreweave_s3()
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = document_rows()
    for verdict in ("kept", "flagged"):
        group = [r for r in rows if (r["category"] == "kept") == (verdict == "kept")]
        specific = [r for r in group if r["specific"]]
        print(
            f"{verdict}: {len(group)} documents, filtered-specific {sum(r['gap'] < 0 for r in specific)} better "
            f"and {sum(r['gap'] > 0 for r in specific)} worse"
        )
        print(f"  all: {group_losses(group)}")
        clean = [r for r in group if r["category"] != "encoded_blob" and not r["breakdown"]]
        print(f"  without encoded blobs and breakdowns ({len(clean)}): {group_losses(clean)}")
        for label, reverse in (("filtered better", False), ("filtered worse", True)):
            print(f"  {label}:")
            for r in sorted(group, key=lambda r: r["gap"], reverse=reverse)[:top]:
                print(
                    f"    {r['cell']} {r['index']} {r['category']} tokens={r['tokens']} base={r['base']:.2f} "
                    f"c0={r['c0']:.2f} c1={r['c1']:.2f} f={r['f']:.2f} gap={r['gap']:+.3f} seed={r['seed']:.3f}"
                    f"{' specific' if r['specific'] else ''}{' breakdown' if r['breakdown'] else ''}"
                )


if __name__ == "__main__":
    main()
