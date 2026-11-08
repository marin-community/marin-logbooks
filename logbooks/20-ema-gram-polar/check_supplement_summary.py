# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise paired signs and host aggregation with known synthetic scalar fixtures."""

import json
import math
import tempfile
from pathlib import Path

from summarize_supplement import paired, summarize


def execute():
    baseline = [3 + i / 100 for i in range(8)]
    losses = []
    for counter in (3733, 18665, 33597):
        methods = [(alpha, "newton_schulz") for alpha in (0.0, 0.25, 0.5)]
        if counter == 18665:
            methods.append((0.25, "svd"))
        for alpha, method in methods:
            for size in (1e-5, 3e-5, 1e-4):
                values = [v + alpha * size + (0.0002 if method == "svd" else 0) for v in baseline]
                delta = paired(values, baseline)
                losses.append(
                    dict(
                        kind="loss",
                        counter=counter,
                        alpha=alpha,
                        method=method,
                        step_size=size,
                        ridge=0.01,
                        losses=values,
                        baseline_losses=baseline,
                        delta_mean=delta["mean"],
                        delta_sem=delta["sem"],
                    )
                )
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for rank in range(4):
            path = Path(tmp) / f"rank-{rank}.jsonl"
            rows = [
                dict(
                    kind="data",
                    batch_offset=8,
                    batches=8,
                    process_count=4,
                    device_count=16,
                    process_index=rank,
                    local_shard_sha256=["fixture"] * 8,
                )
            ]
            rows += [dict(kind="restore", counter=c) for c in (3733, 18665, 33597)]
            rows += losses
            timings = [
                dict(alpha=a, repeat=i, seconds=(1 + a) * (i + 1) + rank) for i in range(4) for a in (0.0, 0.25, 0.5)
            ]
            rows.append(dict(kind="matrix_direction_timings", rows=timings, state_bytes={}))
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            paths.append(path)
        result = summarize(paths)
        assert len(result["norm_contrasts"]) == 18 and len(result["exact_minus_production"]) == 3
        for row in result["norm_contrasts"]:
            assert math.isclose(row["mean"], (0.25 - row["comparator_alpha"]) * row["step_size"], abs_tol=1e-14)
        assert all(math.isclose(row["mean"], 0.0002, abs_tol=1e-14) for row in result["exact_minus_production"])
        assert result["timing_median_max_rank_seconds"] == {"0.0": 5.5, "0.25": 6.125, "0.5": 6.75}
        try:
            summarize(paths[:3])
        except ValueError:
            pass
        else:
            raise AssertionError("Missing host must fail")
        raw = paths[3].read_text()
        rows = [json.loads(line) for line in raw.splitlines()]
        row = next(r for r in rows if r["kind"] == "loss")
        row["losses"] = [x + 0.1 for x in row["losses"]]
        delta = paired(row["losses"], row["baseline_losses"])
        row.update(delta_mean=delta["mean"], delta_sem=delta["sem"])
        paths[3].write_text("".join(json.dumps(r) + "\n" for r in rows))
        try:
            summarize(paths)
        except AssertionError:
            pass
        else:
            raise AssertionError("Host disagreement must fail")
    return dict(
        paired_signs=True,
        svd_contrast=True,
        max_rank_then_median=True,
        missing_host_rejected=True,
        disagreement_rejected=True,
    )


if __name__ == "__main__":
    print(json.dumps(execute()))
