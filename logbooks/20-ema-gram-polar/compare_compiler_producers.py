# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description="Compare captured operand-producing compiler bodies, not runtime behavior.")
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
src = args.report
d = json.loads(src.read_text())
maps = {}
rx = re.compile(r"^\s*(?:ROOT\s+)?%([^ ]+) = (.*?) ([a-z][a-z0-9-]*)\(([^)]*)\)(.*)$")


def sig(body, bodies, visited=()):
    names = {}
    out = []
    for line in body:
        m = rx.match(line)
        if not m:
            continue
        name, shape, op, operands, attrs = m.groups()
        names[name] = len(names)
        args = re.sub(r"%([\w.-]+)", lambda m: "@" + str(names[m[1]]), operands)
        attrs = attrs.split(", metadata=", 1)[0]
        for target in re.findall(r"calls=%([^, }]+)", attrs):
            if target not in bodies or target in visited:
                return None
            nested = sig(bodies[target], bodies, (*visited, target))
            if nested is None:
                return None
            attrs = attrs.replace("calls=%" + target, "calls=" + json.dumps(nested))
        out.append((shape, op, args, attrs))
    return out


for name, a in d["artifacts"].items():
    ops = {r["computation"] + "/" + r["name"]: r for r in a["selected_operations"]}
    maps[name] = {}
    for g, rs in a["polar_lineage"]["output_groups"].items():
        for r in rs:
            op = ops[r["operation"]]
            maps[name][(g, r["iteration"], r["product"])] = [
                (
                    sig(a["producer_bodies"][p["calls"]], a["producer_bodies"], (p["calls"],))
                    if p["calls"]
                    else [("direct", p["opcode"], p["shape"])]
                )
                for p in op["direct_operand_producers"]
            ]
report = {
    "input_report_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    "method": (
        "Compare immediate operand producer instruction signatures, preserving shape/opcode/operand "
        "order/constants/parameter indices/attributes before metadata; rename SSA names by local instruction "
        "order; recursively include available nested fusion bodies; reject missing/cyclic callees. Secondary "
        "comparison removes only S(1) tags."
    ),
    "limitations": (
        "Not whole-graph equivalence, upstream buffer equality, backend configuration, scheduling, runtime "
        "ordering or causal proof. Counts are matched static product positions, not independent runtime "
        "invocations."
    ),
    "comparisons": {},
}
for a, b in [("fused_original", "fused_matrix"), ("direction_original", "direction_matrix")]:
    valid = [k for k in maps[a] if all(x is not None for x in maps[a][k] + maps[b][k])]
    diff = [k for k in valid if maps[a][k] != maps[b][k]]
    remaining = [
        k for k in valid if json.dumps(maps[a][k]).replace("S(1)", "") != json.dumps(maps[b][k]).replace("S(1)", "")
    ]
    report["comparisons"][a + "_vs_" + b] = {
        "compared_product_pairs": len(valid),
        "different_signatures": len(diff),
        "different_after_removing_only_S1": len(remaining),
        "remaining_keys": remaining,
        "excluded_keys": [k for k in maps[a] if k not in valid],
        "first_difference": {"key": diff[0], "left": maps[a][diff[0]], "right": maps[b][diff[0]]} if diff else None,
    }
args.output.write_text(json.dumps(report, indent=2) + "\n")
