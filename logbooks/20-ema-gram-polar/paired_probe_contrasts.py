# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare quarter-power probe losses with paired zero/half-power losses."""

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def main(source: Path, output: Path) -> None:
    raw = source.read_bytes()
    data = json.loads(raw)
    contrasts = []
    summaries = []
    for probe in data["probes"]:
        rows = probe["rows"]
        for row in (r for r in rows if r["alpha"] == 0.25):
            for other in [0.0, 0.5]:
                matches = [
                    r
                    for r in rows
                    if r["alpha"] == other
                    and r["step_size"] == row["step_size"]
                    and (other == 0 or r["ridge"] == row["ridge"])
                ]
                assert len(matches) == 1
                reference = matches[0]
                assert row["baseline_losses"] == reference["baseline_losses"]
                assert len(row["losses"]) == len(reference["losses"]) == 8
                deltas = [a - b for a, b in zip(row["losses"], reference["losses"], strict=True)]
                mean = statistics.mean(deltas)
                assert math.isclose(mean, row["delta_mean"] - reference["delta_mean"], abs_tol=1e-15)
                contrasts.append(
                    {
                        "counter": probe["counter"],
                        "alpha": 0.25,
                        "comparator_alpha": other,
                        "ridge": row["ridge"],
                        "step_size": row["step_size"],
                        "paired_batch_deltas": deltas,
                        "mean": mean,
                        "std": statistics.stdev(deltas),
                        "sem": statistics.stdev(deltas) / math.sqrt(8),
                        "independent_batches": 8,
                    }
                )
        for other in [0.0, 0.5]:
            cells = [r for r in contrasts if r["counter"] == probe["counter"] and r["comparator_alpha"] == other]
            assert len(cells) == 6
            summaries.append(
                {
                    "counter": probe["counter"],
                    "comparator_alpha": other,
                    "negative_mean_cells": sum(r["mean"] < 0 for r in cells),
                    "positive_mean_cells": sum(r["mean"] > 0 for r in cells),
                    "total_cells": len(cells),
                    "mean_delta_range": [min(r["mean"] for r in cells), max(r["mean"] for r in cells)],
                }
            )
    assert len(contrasts) == 36
    result = {
        "status": "CACHED_PAIRED_NORM_MATCHED_CONTRASTS",
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "definition": (
            "Quarter-power perturbed loss minus comparator perturbed loss on the same batch; negative favors 1/4."
        ),
        "cells": contrasts,
        "summary": summaries,
        "limits": [
            "Eight fixed batches; four host replicas are not additional samples; batches reused across stages.",
            "Same checkpoint and step size; same ridge for nonzero comparator. All 36 contrasts retained.",
            "Counts summarize cells, not independent trials or significance tests. No best-cell selection.",
            "SEM is conditional batch uncertainty, not training-seed uncertainty or a full-training effect.",
        ],
        "remote_experiment_reads": 0,
        "new_experiments": 0,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
