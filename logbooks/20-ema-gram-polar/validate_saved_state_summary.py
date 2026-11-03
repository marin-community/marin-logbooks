# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static mutation fixtures for complete distributed saved-state report joins."""

import copy
import hashlib
import json
import math
from pathlib import Path

from summarize_saved_state_check import CASES, SHAPES, expected_records, summarize


def validate():
    base = Path(__file__).parent
    template = json.loads((base / "saved-state-check-cpu-validation.json").read_text())
    manifest = json.loads((base / "saved-state-check-manifest.json").read_text())
    identity = json.loads((base / "loss-pair-checkpoint-metadata.json").read_text())
    reports = []
    for process in range(4):
        r = copy.deepcopy(template)
        r.update(mode="worker", process_index=process, process_count=4, shapes=SHAPES)
        r["gradient_scale"] = 0.1 / math.sqrt(sum(math.prod(s) for s in SHAPES))
        r["checkpoint"].update(
            checkpoint_uri=identity["checkpoint"],
            manifest_sha256=identity["files"]["manifest.json"]["sha256"],
            loaded_bytes=3460300800,
            metadata=identity["metadata"],
            loaded_paths=[entry["path"] for entry in identity["state_arrays"]],
        )
        for case in CASES:
            rows = []
            expected = expected_records(case == CASES[1])
            for (leaf, index), shape in expected.items():
                keys = [key for key in expected if key[0] == leaf]
                if leaf < 7 and keys.index((leaf, index)) // 4 != process:
                    continue
                rows.append(
                    dict(
                        leaf=leaf,
                        index=index,
                        shape=shape,
                        dtype="float32",
                        passed=False,
                        max_absolute_error=0.5,
                        squared_error=0.25,
                        squared_reference=1.0,
                        actual_sha256="a" * 64,
                        reference_sha256="b" * 64,
                    )
                )
            r["cases"][case]["comparison"].update(shards=rows, local_passed=False)
        reports.append(r)
    result = summarize(reports, manifest, identity)
    assert [result["cases"][case]["unique_leaf_index_records"] for case in CASES] == [112, 133]
    assert all(not result["cases"][case]["passed"] for case in CASES)
    assert all(row["relative_frobenius_error"] == 0.5 for case in CASES for row in result["cases"][case]["leaves"])
    assert result["finite_and_immutable_all_processes"] is True
    reordered = copy.deepcopy(reports[::-1])
    for r in reordered:
        for case in CASES:
            r["cases"][case]["comparison"]["shards"].reverse()
    assert summarize(reordered, manifest, identity) == result
    rejected = []

    def rejects(name, change):
        broken = copy.deepcopy(reports)
        change(broken)
        try:
            summarize(broken, manifest, identity)
        except (AssertionError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError(f"Accepted invalid fixture: {name}")

    rejects("missing_process", lambda r: r.pop())
    rejects("duplicate_process", lambda r: r[3].update(process_index=2))
    rejects("unfinished_process", lambda r: r[1].update(completed=False))
    rejects("source_mismatch", lambda r: r[2]["source_files"].clear())
    rejects("checkpoint_mismatch", lambda r: r[3]["checkpoint"].update(checkpoint_uri="wrong"))
    rejects("wrong_saved_paths", lambda r: r[0]["checkpoint"]["loaded_paths"].reverse())
    rejects("wrong_geometry", lambda r: r[0].update(shapes=[]))
    rejects("changed_input", lambda r: r[0]["cases"][CASES[0]]["matrix"].update(input_state_unchanged=False))
    rejects("nonfinite_output", lambda r: r[0]["cases"][CASES[1]]["replicated"].update(all_local_outputs_finite=False))
    rejects("missing_case", lambda r: r[0]["cases"].pop(CASES[0]))
    rejects("missing_shard", lambda r: r[0]["cases"][CASES[0]]["comparison"]["shards"].pop())
    rejects(
        "wrong_index",
        lambda r: r[0]["cases"][CASES[0]]["comparison"]["shards"][0].update(index="wrong"),
    )
    rejects(
        "replica_disagreement",
        lambda r: r[0]["cases"][CASES[1]]["comparison"]["shards"][-1].update(actual_sha256="c" * 64),
    )
    rejects(
        "nonfinite_statistic",
        lambda r: r[0]["cases"][CASES[0]]["comparison"]["shards"][0].update(squared_error=float("nan")),
    )
    rejects("wrong_tolerance", lambda r: r[0]["cases"][CASES[0]]["comparison"].update(atol=1e-4))
    return dict(
        scope="Generated static JSON fixtures only; no new measured numerical result or accelerator execution",
        full_reports_required=4,
        unique_record_counts=[112, 133],
        reference_norm_fixture=0.5,
        finite_equivalence_failure_retained_without_veto=True,
        report_and_row_order_invariance=True,
        rejected_mutations=rejected,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), base / "summarize_saved_state_check.py")
        },
    )


if __name__ == "__main__":
    output = Path(__file__).with_name("saved-state-summary-validation.json")
    output.write_text(json.dumps(validate(), indent=2) + "\n")
    print(output)
