# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the selected input of down normalization's declared combined all-reduce."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from compare_outer_bindings import normalized
from compare_start_producers import canonical_trace
from consolidate_compiler_records import identity


def summarize(report):
    assert report["completed"]
    result = {}
    for name, graph in report["artifacts"].items():
        assert graph["sha256_verified"]
        target = graph["trace"]["targets"]["down_proj/opaque/ConcatBitcast"]
        assert sorted(o["operand_position"] for o in target["operands"]) == [0, 1, 2, 3]
        rows = []
        for operand in sorted(target["operands"], key=lambda o: o["operand_position"]):
            value = operand["trace"]
            matches = [(i, r) for i, r in enumerate(value["records"]) if r["opcode"] == "all-reduce"]
            assert len(matches) == 1
            index, row = matches[0]
            assert len(row["tuple_path"]) == 1
            position = row["tuple_path"][0]
            outgoing = [e for e in value["edges"] if e["source"] == index]
            assert len(outgoing) == 1 and outgoing[0]["operand_index"] == position
            incoming = [e for e in value["edges"] if e["target"] == index]
            assert len(incoming) == 1
            projection = value["records"][incoming[0]["source"]]
            assert projection["opcode"] == "get-tuple-element"
            assert re.search(r"index=(\d+)", projection["attributes"])[1] == str(position)
            assert "/vmap(jit(norm))/reduce_sum" in projection["attributes"]
            child = outgoing[0]["target"]
            # A separate rooted comparison preserves the original all-reduce projection mismatch.
            signature = normalized(dict(canonical_trace(value), root=child), name)[0]
            rows.append(
                dict(
                    operand_position=operand["operand_position"],
                    all_reduce=row,
                    projection=projection,
                    selected_input_root=value["records"][child],
                    selected_input_signature=signature,
                )
            )
        assert len({identity(r["all_reduce"]) for r in rows}) == 1
        assert len({r["selected_input_signature"] for r in rows}) == 1
        result[name] = rows
    base = result["matrix_untouched"]
    for rows in result.values():
        for row, reference in zip(rows, base, strict=True):
            row["same_selected_input_signature_as_untouched"] = (
                row["selected_input_signature"] == reference["selected_input_signature"]
            )
    return dict(
        graphs=result,
        limitations=[
            "Four views share one exact all-reduce projection per graph; no ordinal or text-order matching.",
            "Selected-input subtree comparison is separate and does not erase whole-route mismatches.",
            "Bounded declarations and connections only; no runtime values, precision, order or cause.",
            "All custom-call outputs remain opaque and both full28leaf numerical gates remain failed.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    args.output.write_text(json.dumps(summarize(json.loads(raw)), indent=2) + "\n")


if __name__ == "__main__":
    main()
