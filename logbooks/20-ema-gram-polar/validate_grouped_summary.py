# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static fixtures for the grouped-Gram diagnostic's distributed report summary."""

import copy
import hashlib
import json
import math
from pathlib import Path

from summarize_grouped_gram import CASES, LAYOUTS, summarize
from summarize_saved_state_check import SHAPES, expected_records


def validate():
    base = Path(__file__).parent
    template = json.loads((base / "grouped-gram-cpu-validation.json").read_text())
    manifest = json.loads((base / "grouped-gram-manifest.json").read_text())
    identity = json.loads((base / "loss-pair-checkpoint-metadata.json").read_text())
    commit = json.loads((base / "grouped-gram-submitted.json").read_text())["source_commit"]
    reports = []
    for process in range(4):
        r = copy.deepcopy(template)
        r.update(mode="worker", process_index=process, process_count=4, shapes=SHAPES, source_commit=commit)
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
            for layout, scale in zip(LAYOUTS, (2, 1), strict=True):
                r["cases"][case][layout].update(
                    compile_and_first_seconds=10 * scale,
                    warmed_sample_seconds=[scale, 2 * scale, 3 * scale],
                    bitwise_replays=2,
                    replay_comparisons=[True, False, True],
                )
        reports.append(r)
    result = summarize(reports, manifest, identity, commit)
    assert [result["cases"][case]["unique_leaf_index_records"] for case in CASES] == [112, 133]
    for case in CASES:
        assert result["cases"][case]["passed"] is False
        assert all(row["relative_frobenius_error"] == 0.5 for row in result["cases"][case]["leaves"])
        for timing in result["cases"][case]["timing_by_process"].values():
            assert timing["baseline_over_candidate_mean_ratio"] == 2
            assert timing["matrix"]["mean_seconds"] == 4 and timing["matrix_grouped"]["mean_seconds"] == 2
            assert timing["matrix_grouped"]["bitwise_replays"] == 2
    reordered = copy.deepcopy(reports[::-1])
    for r in reordered:
        for case in CASES:
            r["cases"][case]["comparison"]["shards"].reverse()
    assert summarize(reordered, manifest, identity, commit) == result
    rejected = []

    def rejects(name, change):
        broken = copy.deepcopy(reports)
        change(broken)
        try:
            summarize(broken, manifest, identity, commit)
        except (AssertionError, KeyError, TypeError):
            rejected.append(name)
        else:
            raise AssertionError(f"Accepted invalid fixture: {name}")

    rejects("missing_process", lambda r: r.pop())
    rejects("duplicate_process", lambda r: r[3].update(process_index=2))
    rejects("unfinished_process", lambda r: r[1].update(completed=False))
    rejects("source_mismatch", lambda r: r[2]["source_files"].clear())
    rejects("commit_mismatch", lambda r: r[0].update(source_commit="0" * 40))
    rejects("checkpoint_mismatch", lambda r: r[3]["checkpoint"].update(checkpoint_uri="wrong"))
    rejects("wrong_saved_paths", lambda r: r[0]["checkpoint"]["loaded_paths"].reverse())
    rejects("wrong_geometry", lambda r: r[0].update(shapes=[]))
    rejects("changed_input", lambda r: r[0]["cases"][CASES[0]]["matrix"].update(input_immutable=False))
    rejects("nonfinite_output", lambda r: r[0]["cases"][CASES[1]]["matrix_grouped"].update(finite=False))
    rejects("wrong_layout", lambda r: r[0].update(candidate="matrix"))
    rejects("missing_case", lambda r: r[0]["cases"].pop(CASES[0]))
    rejects("missing_shard", lambda r: r[0]["cases"][CASES[0]]["comparison"]["shards"].pop())
    rejects("wrong_index", lambda r: r[0]["cases"][CASES[0]]["comparison"]["shards"][0].update(index="wrong"))
    rejects(
        "replica_disagreement",
        lambda r: r[0]["cases"][CASES[1]]["comparison"]["shards"][-1].update(actual_sha256="c" * 64),
    )
    rejects(
        "nonfinite_statistic",
        lambda r: r[0]["cases"][CASES[0]]["comparison"]["shards"][0].update(squared_error=float("nan")),
    )
    rejects("wrong_tolerance", lambda r: r[0]["cases"][CASES[0]]["comparison"].update(atol=1e-4))
    rejects("wrong_sample_count", lambda r: r[0]["cases"][CASES[0]]["matrix"]["warmed_sample_seconds"].pop())
    rejects("negative_time", lambda r: r[0]["cases"][CASES[0]]["matrix"].update(compile_and_first_seconds=-1))
    rejects(
        "nonfinite_time", lambda r: r[0]["cases"][CASES[0]]["matrix"].update(warmed_sample_seconds=[1, 2, float("inf")])
    )
    rejects("replay_count_mismatch", lambda r: r[0]["cases"][CASES[0]]["matrix"].update(bitwise_replays=3))
    rejects("replay_wrong_type", lambda r: r[0]["cases"][CASES[0]]["matrix"].update(replay_comparisons=[True, 0, True]))
    rejects("training_steps", lambda r: r[0].update(training_steps=1))
    rejects("full_restore_claim", lambda r: r[0].update(full_trainer_restore=True))
    return dict(
        scope="Generated static JSON fixtures, no kernels, new compute job or measured TPU result",
        unique_record_counts=[112, 133],
        reference_relative_norm=0.5,
        reference_optimizer_mean_ratio=2,
        finite_tolerance_failure_and_nonbitwise_replays_retained=True,
        report_row_order_invariance=True,
        rejected_mutations=rejected,
        external_checks_required=["canonical success", "same-attempt identity", "fresh online logging"],
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                base / "summarize_grouped_gram.py",
                base / "summarize_saved_state_check.py",
            )
        },
    )


if __name__ == "__main__":
    path = Path(__file__).with_name("grouped-gram-summary-validation.json")
    path.write_text(json.dumps(validate(), indent=2) + "\n")
    print(path)
