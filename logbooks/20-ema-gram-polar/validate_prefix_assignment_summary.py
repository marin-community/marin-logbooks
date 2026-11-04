# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mutation checks for the bounded assignment joins, without numerical kernels."""

import copy
import json
from pathlib import Path

from summarize_prefix_assignment import summarize

ROOT = Path(__file__).resolve().parent


def main():
    report = json.loads(Path("scratch/20261104-prefix-assignment-label-final.json").read_text())
    manifest = json.loads((ROOT / "prefix-assignment-manifest.json").read_text())
    prior = json.loads((ROOT / "prefix-core-metadata-summary.json").read_text())
    intervals = json.loads(Path("scratch/20261104-prefix-intervals-final.json").read_text())
    expected = summarize(report, manifest, prior, intervals)
    assert expected["module_count"] == 36 and len(expected["slice_routes"]) == 14
    reordered = copy.deepcopy(report)
    reordered["hosts"].reverse()
    for host in reordered["hosts"]:
        summary = host["summary"]
        for key in ["executables", "cores", "assignment_events", "planes"]:
            summary[key].reverse()
        for module in summary["executables"]:
            module["hlo"]["selected_computations"].reverse()
            module["hlo"]["direct_callers"].reverse()
            for comp in module["hlo"]["selected_computations"]:
                comp["instructions"].reverse()
    assert summarize(reordered, manifest, prior, intervals) == expected

    def mutate(data, kind):
        host = data["hosts"][0]
        summary = host["summary"]
        train = next(m["hlo"] for m in summary["executables"] if m["hlo"]["module_name"] == "jit__train_step")
        event = next(e for e in summary["assignment_events"] if "(jit__train_step)" in e["event_name"])
        slice_comp = next(
            c
            for c in train["selected_computations"]
            if any(i["opcode"] == "dynamic-slice" and i["id"] in c["target_ids"] for i in c["instructions"])
        )
        if kind == "profile_sha":
            host["verified_identity"]["sha256"] = "bad"
        elif kind == "missing_host":
            data["hosts"].pop()
        elif kind == "missing_executable":
            summary["executables"].pop()
        elif kind == "duplicate_executable":
            summary["executables"].append(copy.deepcopy(summary["executables"][0]))
        elif kind == "assignment_swap":
            train["computation_devices"][8:10] = list(reversed(train["computation_devices"][8:10]))
        elif kind == "event_wrong_device":
            next(s for s in event["stats"] if s["name"] == "global_device_id")["value"] = 99
        elif kind == "event_missing":
            summary["assignment_events"].remove(event)
        elif kind == "core_payload":
            summary["cores"][0]["payload_hex"] = "00"
        elif kind == "core_plane":
            summary["cores"][0]["plane_name"] = "wrong"
        elif kind == "slice_axis_swap":
            leaf = next(i for i in slice_comp["instructions"] if i["id"] in slice_comp["target_ids"])
            leaf["operand_ids"][1:3] = list(reversed(leaf["operand_ids"][1:3]))
        elif kind == "wrong_caller":
            next(c for c in train["direct_callers"] if slice_comp["id"] in c["instruction"]["called_computation_ids"])[
                "computation_id"
            ] = -1
        elif kind == "nonzero_axis":
            next(i for i in slice_comp["instructions"] if "s32s" in i)["s32s"] = [1]
        else:
            raise ValueError(kind)

    kinds = [
        "profile_sha",
        "missing_host",
        "missing_executable",
        "duplicate_executable",
        "assignment_swap",
        "event_wrong_device",
        "event_missing",
        "core_payload",
        "core_plane",
        "slice_axis_swap",
        "wrong_caller",
        "nonzero_axis",
    ]
    for kind in kinds:
        changed = copy.deepcopy(report)
        mutate(changed, kind)
        try:
            summarize(changed, manifest, prior, intervals)
        except (AssertionError, KeyError):
            continue
        raise AssertionError(f"mutation accepted: {kind}")
    result = dict(
        passed=True,
        mutation_rejections=kinds,
        host_module_instruction_order_invariance=True,
        actual_module_records=36,
        actual_slice_routes=14,
        numerical_kernels=False,
        raw_hlo_equal_claim=False,
        missing_pad_configuration_preserved=True,
    )
    (ROOT / "prefix-assignment-summary-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
