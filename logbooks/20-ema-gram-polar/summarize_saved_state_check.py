# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join bounded saved-state reports without importing numerical runtimes."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

CASES = ("unchanged_saved_state_direction", "fixed_synthetic_gradient_update")
NAMES = ("q", "k", "v", "o", "gate", "up", "down")
SHAPES = [[11, 1024, 1024]] * 4 + [[11, 4096, 1024]] * 2 + [[11, 1024, 4096]]


def expected_records(full, devices=16):
    """Exact global slices for direction sharding over all selected devices."""
    result = {}
    for leaf, shape in enumerate(SHAPES):
        axis = 1 if leaf in (3, 6) else 2
        width = shape[axis] // devices
        for device in range(devices):
            index = [slice(None)] * 3
            index[axis] = slice(device * width, (device + 1) * width)
            local_shape = list(shape)
            local_shape[axis] = width
            result[leaf, str(tuple(index))] = local_shape
    if full:
        for group, (layers, rows, columns) in enumerate(SHAPES):
            for field, shape in enumerate(([layers, rows, columns], [layers, rows, rows], [layers, columns, columns])):
                result[7 + 3 * group + field, str((slice(None),) * 3)] = shape
    return result


def summarize(reports, manifest, checkpoint_identity, *, processes=4, devices=16):
    assert devices == processes * 4 and devices in (16, 32, 64)
    assert len(reports) == processes, "Require all process reports"
    assert sorted(r["process_index"] for r in reports) == list(range(processes)), "Missing/duplicate process"
    checkpoint = reports[0]["checkpoint"]
    assert checkpoint["checkpoint_uri"] == checkpoint_identity["checkpoint"]
    assert checkpoint["manifest_sha256"] == checkpoint_identity["files"]["manifest.json"]["sha256"]
    assert checkpoint["loaded_bytes"] == 3460300800
    assert len(checkpoint["loaded_paths"]) == len(set(checkpoint["loaded_paths"])) == 21
    assert checkpoint["loaded_paths"] == [entry["path"] for entry in checkpoint_identity["state_arrays"]]
    assert checkpoint["metadata"] == checkpoint_identity["metadata"]
    for r in reports:
        assert r["mode"] == "worker" and r["completed"] is True and r["process_count"] == processes
        assert r["source_files"] == manifest["files"]
        assert r["checkpoint"] == checkpoint and r["shapes"] == SHAPES
        assert r["gradient_seed"] == 0
        assert r["gradient_scale"] == 0.1 / math.sqrt(sum(math.prod(s) for s in SHAPES))
        assert r["input_state_finite"] is True and r["training_steps_executed"] == 0
        assert r["full_training_memory_fit_verified"] is False and r["full_trainer_restore"] is False
        assert r["equivalence_is_training_veto"] is False
        assert set(r["cases"]) == set(CASES)
    output = {}
    for case in CASES:
        full = case == CASES[1]
        expected = expected_records(full, devices)
        joined = {}
        owners = {}
        times = {}
        for r in reports:
            case_report = r["cases"][case]
            for layout in ("replicated", "matrix"):
                item = case_report[layout]
                assert item["all_local_outputs_finite"] is True and item["input_state_unchanged"] is True
                elapsed = item["seconds_including_compilation"]
                assert math.isfinite(elapsed) and elapsed >= 0
            times[str(r["process_index"])] = {
                layout: case_report[layout]["seconds_including_compilation"] for layout in ("replicated", "matrix")
            }
            comparison = case_report["comparison"]
            assert comparison["rtol"] == comparison["atol"] == 1e-5
            rows = comparison["shards"]
            assert len(rows) == (49 if full else 28), "Missing or repeated local output shard"
            assert comparison["local_passed"] == all(row["passed"] for row in rows)
            local_keys = set()
            for row in rows:
                key = row["leaf"], row["index"]
                assert key not in local_keys and key in expected
                local_keys.add(key)
                assert row["shape"] == expected[key] and row["dtype"] == "float32"
                assert isinstance(row["passed"], bool)
                for field in ("max_absolute_error", "squared_error", "squared_reference"):
                    assert math.isfinite(row[field]) and row[field] >= 0
                for field in ("actual_sha256", "reference_sha256"):
                    assert re.fullmatch(r"[0-9a-f]{64}", row[field])
                if key in joined:
                    assert joined[key] == row, "Replicated output hash/statistic disagreement"
                joined[key] = row
                owners.setdefault(key, []).append(r["process_index"])
            assert all(sum(k[0] == leaf for k in local_keys) == 4 for leaf in range(7))
            if full:
                assert all(sum(k[0] == leaf for k in local_keys) == 1 for leaf in range(7, 28))
        assert set(joined) == set(expected), "Incomplete global shard coverage"
        assert all(len(owners[k]) == (1 if k[0] < 7 else processes) for k in joined)
        leaves = []
        for leaf in range(28 if full else 7):
            rows = [row for (index, _), row in joined.items() if index == leaf]
            error = sum(row["squared_error"] for row in rows)
            reference = sum(row["squared_reference"] for row in rows)
            assert math.isfinite(error) and math.isfinite(reference), "Nonfinite global norm accumulation"
            label = (
                f"direction/{NAMES[leaf]}"
                if leaf < 7
                else f"state/{NAMES[(leaf - 7) // 3]}/{('momentum', 'left', 'right')[(leaf - 7) % 3]}"
            )
            leaves.append(
                dict(
                    leaf=leaf,
                    label=label,
                    unique_shards=len(rows),
                    passed=all(row["passed"] for row in rows),
                    max_absolute_error=max(row["max_absolute_error"] for row in rows),
                    relative_frobenius_error=math.sqrt(error) / max(math.sqrt(reference), 1e-30),
                    all_recorded_shard_hashes_equal=all(row["actual_sha256"] == row["reference_sha256"] for row in rows),
                )
            )
        output[case] = dict(
            rtol=1e-5,
            atol=1e-5,
            passed=all(row["passed"] for row in leaves),
            leaves=leaves,
            unique_leaf_index_records=len(joined),
            seconds_including_compilation_by_process=times,
            timing_scope="Compilation-inclusive diagnostic execution, not steady training throughput",
        )
    return dict(
        process_indices=list(range(processes)),
        checkpoint=checkpoint,
        source_files=manifest["files"],
        cases=output,
        finite_and_immutable_all_processes=True,
        full_training_memory_fit_verified=False,
        training_steps_executed=0,
        equivalence_is_training_veto=False,
        limitations=(
            "Join checks report completeness and consistency; "
            "canonical job success and online logging must be verified separately"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).parent
    result = summarize(
        [json.loads(path.read_text()) for path in args.reports],
        json.loads((base / "saved-state-check-manifest.json").read_text()),
        json.loads((base / "loss-pair-checkpoint-metadata.json").read_text()),
    )
    result["input_reports"] = [
        dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in args.reports
    ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
