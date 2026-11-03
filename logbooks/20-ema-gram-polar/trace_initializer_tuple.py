# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace declared whole-tuple initializers; preserve loop and custom-call output opacity."""

import hashlib
import json
import re

from compiler_operation_inventory import CALLS, TUPLE_INDEX

OPAQUE = {
    "custom-call",
    "while",
    "call",
    "conditional",
    "async-start",
    "async-done",
    "async-collective-start",
    "async-collective-done",
}


def trace(scopes, scope, name, bindings, context, polar):
    records, edges, missing, boundaries, memo = [], [], set(), [], {}

    def walk(scope, name, bindings, context, path=()):
        key = (context, scope, name, path)
        if key in memo:
            return memo[key]
        index = len(records)
        memo[key] = index
        if scope not in scopes or name not in scopes[scope][0]:
            missing.add(scope + "/" + name)
            records.append(dict(computation=scope, name=name, context=context, tuple_path=path, missing=True))
            return index
        node = scopes[scope][0][name]
        records.append(dict(computation=scope, context=context, tuple_path=path, **node))
        op, operands = node["opcode"], node["operands"]
        identity = scope + "/" + name

        def follow(position, s=scope, n=None, b=bindings, c=context, p=()):
            child = walk(s, operands[position] if n is None else n, b, c, p)
            edges.append(dict(source=index, operand_index=position, target=child))

        if identity in polar:
            boundaries.append(dict(node=index, reason="polar product anchor", role=polar[identity]))
        elif op == "parameter":
            if bindings is None:
                boundaries.append(dict(node=index, reason="entry parameter"))
            else:
                s, n, b, c = bindings[node["parameter_index"]]
                follow(node["parameter_index"], s, n, b, c, path)
        elif op == "get-tuple-element":
            follow(0, p=(int(TUPLE_INDEX.search(node["attributes"])[1]), *path))
        elif op == "tuple":
            if path:
                follow(path[0], p=path[1:])
            else:
                for position in range(len(operands)):
                    follow(position)
        elif op == "fusion":
            callee = CALLS.search(node["attributes"])[1]
            if callee not in scopes:
                missing.add(callee)
            else:
                args = [(scope, n, bindings, context) for n in operands]
                follow(0, callee, scopes[callee][1], args, (*context, identity), path)
        elif op == "all-reduce" and len(operands) > 1:
            if path:
                follow(path[0], p=path[1:])
            else:
                boundaries.append(dict(node=index, reason="unselected whole collective output"))
        elif op == "copy-done":
            follow(0, p=(0, *path))
        elif op == "copy-start":
            assert path and path[0] in (0, 1)
            follow(0, p=path[1:])
        elif op == "slice-done":
            assert not path and scopes[scope][0][operands[0]]["opcode"] == "slice-start"
            follow(0)
        elif op == "slice-start":
            assert not path and len(operands) == 1
            follow(0)
        elif op in OPAQUE or path:
            boundaries.append(dict(node=index, reason="opaque operation or unsupported tuple projection"))
        else:
            for i in range(len(operands)):
                follow(i)
        return index

    root = walk(scope, name, bindings, context)
    children = {}
    for edge in edges:
        children.setdefault(edge["source"], []).append((edge["operand_index"], edge["target"]))
    labels = {b["node"]: b for b in boundaries}
    hashes = {}

    def signature(index):
        if index in hashes:
            return hashes[index]
        row = records[index]
        attributes = re.sub(r",? metadata=\{[^}]*\}", "", row.get("attributes", ""))
        attributes = re.sub(r"%[A-Za-z0-9_.-]+", "<identifier>", attributes)
        fields = dict(
            shape=row.get("shape"),
            opcode=row.get("opcode"),
            attributes=attributes,
            tuple_path=row["tuple_path"],
            missing=row.get("missing", False),
        )
        if index in labels:
            fields["boundary"] = {k: v for k, v in labels[index].items() if k != "node"}
            if labels[index]["reason"] == "entry parameter":
                fields["source_label"] = re.findall(r'op_name="([^"]+)"', row["attributes"])
                fields["parameter_index"] = row["parameter_index"]
        fields["ordered_edges"] = [(i, signature(j)) for i, j in sorted(children.get(index, []))]
        hashes[index] = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
        return hashes[index]

    return dict(
        root=root,
        records=records,
        edges=edges,
        missing_records=sorted(missing),
        boundaries=boundaries,
        rooted_signature=signature(root),
        signature_scope=(
            "Ordered rooted declarations/connections to boundaries; identifiers/metadata removed except entry labels. "
            "Not runtime values or complete semantic identity; sharing is retained in raw edges "
            "but not distinguished by the recursive signature."
        ),
    )
