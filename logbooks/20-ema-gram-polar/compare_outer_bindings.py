# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare bounded rooted declarations and connections; retain boundary records."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from compiler_operation_inventory import INSTRUCTION


def source_role(label, variant):
    if variant == "matrix_untouched":
        assert re.fullmatch(r"gradients\[\d+\]|states\[\d+\]\.(momentum|left|right)", label)
        return label
    prefix = "bank[0]" if variant == "matrix_substitute" else "bank"
    if label.startswith(prefix + "[0]"):
        return "gradients" + label[len(prefix + "[0]") :]
    assert label.startswith(prefix + "[1]")
    return "states" + label[len(prefix + "[1]") :]


def node_fields(row, boundary, variant):
    attributes = re.sub(r",? metadata=\{[^}]*\}", "", row.get("attributes", ""))
    attributes = re.sub(r"%[A-Za-z0-9_.-]+", "<identifier>", attributes)
    result = dict(shape=row.get("shape"), opcode=row.get("opcode"), attributes=attributes, tuple_path=row["tuple_path"])
    if row.get("opcode") == "constant":
        result["literal"] = INSTRUCTION.match(row["instruction"])[4]
    if boundary:
        result["boundary"] = {k: v for k, v in boundary.items() if k != "node"}
        if boundary["reason"] == "entry parameter":
            labels = re.findall(r'op_name="([^"]+)"', row["attributes"])
            assert len(labels) == 1
            result["source_role"] = source_role(labels[0], variant)
    return result


def normalized(value, variant):
    assert not value["missing_records"]
    boundary = {b["node"]: b for b in value["boundaries"]}
    fields = [node_fields(row, boundary.get(i), variant) for i, row in enumerate(value["records"])]
    children = {}
    for edge in value["edges"]:
        children.setdefault(edge["source"], []).append((edge["operand_index"], edge["target"]))
    hashes = {}

    def digest(index):
        if index not in hashes:
            obj = dict(fields=fields[index], edges=[(i, digest(j)) for i, j in sorted(children.get(index, []))])
            hashes[index] = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
        return hashes[index]

    return digest(value["root"]), fields, children


def rooted_differences(left, right, variant):
    _, lf, lc = normalized(left, "matrix_untouched")
    _, rf, rc = normalized(right, variant)
    seen, differences = set(), []

    def visit(i, j, path):
        if (i, j) in seen:
            return
        seen.add((i, j))
        changed = [k for k in sorted(set(lf[i]) | set(rf[j])) if lf[i].get(k) != rf[j].get(k)]
        a, b = sorted(lc.get(i, [])), sorted(rc.get(j, []))
        edge_mismatch = [p for p, _ in a] != [p for p, _ in b]
        stop = lf[i]["opcode"] != rf[j]["opcode"] or edge_mismatch
        if changed or edge_mismatch:
            differences.append(
                dict(
                    operand_path=path,
                    left_node=i,
                    right_node=j,
                    changed_fields=changed,
                    left={k: lf[i].get(k) for k in changed},
                    right={k: rf[j].get(k) for k in changed},
                    operand_positions_differ=edge_mismatch,
                    unaligned_subtrees_not_compared=stop,
                )
            )
        if not stop:
            for (p, k), (_, l) in zip(a, b, strict=True):
                visit(k, l, [*path, p])

    visit(left["root"], right["root"], [])
    return differences


def summarize(value, variant):
    signature, fields, _ = normalized(value, variant)
    return dict(
        normalized_signature=signature,
        raw_signature=value["rooted_signature"],
        records=len(value["records"]),
        edges=len(value["edges"]),
        boundaries=[dict(node=b["node"], fields=fields[b["node"]]) for b in value["boundaries"]],
        products=[
            dict(
                index=i,
                shape=row["shape"],
                opcode=row["opcode"],
                source_scope=re.findall(r'op_name="([^"]+)"', row["attributes"]),
                operand_edges=[
                    dict(
                        **e,
                        operand_shape=value["records"][e["target"]]["shape"],
                        operand_opcode=value["records"][e["target"]]["opcode"],
                    )
                    for e in value["edges"]
                    if e["source"] == i
                ],
            )
            for i, row in enumerate(value["records"])
            if row.get("opcode") in ("convolution", "dot")
        ],
    )


def compare(report):
    assert report["completed"]
    graphs = report["artifacts"]
    assert all(g["sha256_verified"] for g in graphs.values())
    reference = graphs["matrix_untouched"]["trace"]
    result = {}
    for name, graph in graphs.items():
        result[name] = {}
        for group, roles in graph["trace"].items():
            result[name][group] = {}
            for role, value in roles.items():
                result[name][group][role] = dict(
                    same_normalized_rooted_signature_as_untouched=(
                        normalized(value, name)[0] == normalized(reference[group][role], "matrix_untouched")[0]
                    ),
                    rooted_differences=rooted_differences(reference[group][role], value, name),
                    **summarize(value, name),
                )
    return dict(
        graphs=result,
        limitations=[
            "Signatures compare ordered declarations and connections only to recorded boundaries.",
            "Source roles canonicalized from wrapper argument structure, not surviving parameter ordinals.",
            "Constant literals included; raw signature also retained but not used for comparison.",
            "Opaque boundaries are not traversed; equal signatures do not establish values or semantic identity.",
            "Both numerical full-update substitution gates remain failed; no runtime precision, order or cause claim.",
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
    result = compare(json.loads(raw))
    result["report_sha256"] = args.sha256
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
