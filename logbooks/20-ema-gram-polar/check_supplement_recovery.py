# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check mixed-provenance aggregation with known late losses and rejection controls."""

import copy
import json
import math
from pathlib import Path

from analyze_supplement import analyze
from summarize_supplement import paired


def execute():
    partial = json.loads(Path(__file__).with_name("supplemental-probe-r03-partial-rows.json").read_text())
    losses = []
    for alpha in (0.0, 0.25, 0.5):
        for size in (1e-5, 3e-5, 1e-4):
            baseline = [3 + i / 10 for i in range(8)]
            values = [v + alpha * size for v in baseline]
            delta = paired(values, baseline)
            losses.append(
                dict(
                    kind="loss",
                    counter=33597,
                    alpha=alpha,
                    method="newton_schulz",
                    ridge=0.01,
                    step_size=size,
                    baseline_losses=baseline,
                    losses=values,
                    delta_mean=delta["mean"],
                    delta_sem=delta["sem"],
                )
            )
    order = [(0, 0.0), (0, 0.25), (0, 0.5), (1, 0.5), (1, 0.25), (1, 0.0)]
    late = []
    for rank in range(4):
        data = dict(partial[0], process_index=rank)
        timing = dict(
            kind="matrix_direction_timings",
            counter=33597,
            scope="fixture",
            state_bytes={},
            rows=[dict(repeat=i, alpha=a, seconds=1 + rank + a) for i, a in order],
        )
        late.append([data, dict(kind="restore", counter=33597), *copy.deepcopy(losses), timing])
    result = analyze(partial, late)
    assert result["status"] == "RECOVERED_EARLY_MIDDLE_PLUS_VALIDATED_LATE"
    assert result["timing_median_max_rank_seconds"] == {"0.0": 4.0, "0.25": 4.25, "0.5": 4.5}
    for row in result["norm_contrasts"]:
        if row["counter"] == 33597:
            assert math.isclose(row["mean"], (0.25 - row["comparator_alpha"]) * row["step_size"], abs_tol=1e-14)
            assert row["provenance"] == "validated_late"
        else:
            assert row["provenance"] == "recovered_failed_run"
    for failure in ("missing_host", "shard_mismatch", "replica_disagreement", "missing_repeat"):
        changed = copy.deepcopy(late)
        if failure == "missing_host":
            changed.pop()
        elif failure == "shard_mismatch":
            changed[0][0]["local_shard_sha256"][0] = "different"
        elif failure == "replica_disagreement":
            row = changed[3][2]
            row["losses"] = [x + 0.1 for x in row["losses"]]
            delta = paired(row["losses"], row["baseline_losses"])
            row.update(delta_mean=delta["mean"], delta_sem=delta["sem"])
        else:
            changed[0][-1]["rows"].pop()
        try:
            analyze(partial, changed)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Failed to reject {failure}")
    return dict(
        mixed_provenance=True,
        paired_signs=True,
        host_maximum=True,
        rejected=["missing_host", "shard_mismatch", "replica_disagreement", "missing_repeat"],
    )


if __name__ == "__main__":
    print(json.dumps(execute()))
