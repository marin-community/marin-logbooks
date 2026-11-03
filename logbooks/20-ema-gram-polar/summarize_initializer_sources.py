# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain whole-initializer positions, source boundaries and bounded substitution differences."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_state_initializer import compare, indexed
from consolidate_compiler_records import identity


def summarize(report):
    comparison = compare(report)
    result = {}
    for name, graph in report["artifacts"].items():
        values = indexed(graph)
        key = "down_proj/source/relative_power/while_initializer/0"
        value = values[key]
        root = value["records"][value["root"]]
        assert root["opcode"] == "tuple" and len(root["operands"]) == 13 and not root["tuple_path"]
        children = {}
        for edge in value["edges"]:
            children.setdefault(edge["source"], []).append(edge)
        positions = sorted(children[value["root"]], key=lambda e: e["operand_index"])
        assert [e["operand_index"] for e in positions] == list(range(13))
        assert positions[5]["target"] == positions[7]["target"]
        rows = []
        for edge in positions:
            seen, pending = set(), [edge["target"]]
            while pending:
                node = pending.pop()
                if node in seen:
                    continue
                seen.add(node)
                pending.extend(e["target"] for e in children.get(node, []))
            boundaries = [value["records"][b["node"]] for b in value["boundaries"] if b["node"] in seen]
            rows.append(
                dict(position=edge["operand_index"], producer=value["records"][edge["target"]], boundaries=boundaries)
            )
        assert [i for i, r in enumerate(rows) if r["boundaries"]] == [3, 6, 12]
        state = [r for r in rows[3]["boundaries"] if r["opcode"] == "parameter"]
        product = [r for r in rows[3]["boundaries"] if r["opcode"] in ("dot", "convolution")]
        assert len(state) == len(product) == 1
        state_label = {
            "matrix_untouched": "states[6].right",
            "matrix_evaluate": "bank[1][6].right",
            "matrix_capture": "bank[1][6].right",
            "matrix_substitute": "bank[0][1][6].right",
        }[name]
        assert 'op_name="' + state_label + '"' in state[0]["attributes"]
        assert '/update_gram_state/dot_general"' in product[0]["attributes"]
        assert len(rows[6]["boundaries"]) == len(rows[12]["boundaries"]) == 1
        loop = rows[6]["boundaries"][0]
        assert loop["opcode"] == "while" and loop["tuple_path"] == [6]
        assert identity(loop) == identity(rows[12]["boundaries"][0])
        differences = comparison["graphs"][name][key]["rooted_differences"]
        assert len(differences) == (1 if name == "matrix_substitute" else 0)
        if differences:
            d = differences[0]
            assert d["changed_fields"] == ["attributes"] and not d["unaligned_subtrees_not_compared"]
            assert d["left"]["attributes"].replace("channel_id=329", "channel_id=326") == d["right"]["attributes"]
        result[name] = dict(
            original_while_endpoint=graph["trace"]["targets"][key.rsplit("/", 1)[0]]["endpoint"],
            initializer=root,
            positions=rows,
            repeated_positions=[5, 7],
            shared_opaque_while_positions=[6, 12],
            comparison=comparison["graphs"][name][key],
        )
    return dict(
        graphs=result,
        limitations=[
            "Declared whole initializer only; original while output projection[1] remains opaque.",
            "Positions6/12 reach one earlier while projection[6]; no body or recurrence semantics inferred.",
            "Tuple constituents retain explicit ordered repeated positions and exact caller identities.",
            "No runtime values, precision, order, input equality, semantic identity or cause claim.",
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
