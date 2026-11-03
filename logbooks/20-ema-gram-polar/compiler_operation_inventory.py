# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize existing regional compiler artifacts without executing model kernels."""

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections import Counter
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?%([^ ]+) = (.*?) ([a-z][a-z0-9-]*)\(([^)]*)\)(.*)$")
COMPUTATION = re.compile(r"^(?:ENTRY )?%([^ ]+) .*\{$")
OP_NAME = re.compile(r'metadata=\{op_name="([^"]+)"')
CALLS = re.compile(r"calls=%([^, }]+)")
OPERAND = re.compile(r"%([a-zA-Z0-9_.-]+)")
STACK_FRAME = re.compile(r"stack_frame_id=(\d+)")


def inventory(lines):
    """Retain direct producers of tagged polar and Gram-refresh matrix operations."""
    counts = Counter()
    operations = []
    nodes, selected = {}, []
    scope, total_lines = None, 0
    for number, line in enumerate(lines, 1):
        total_lines = number
        header = COMPUTATION.match(line.rstrip())
        if header:
            assert not nodes, "Unclosed computation"
            scope = header[1]
        if line.strip() == "}":
            for node in selected:
                node["direct_operand_producers"] = [
                    {k: nodes[name][k] for k in ("name", "shape", "opcode", "calls")} for name in node["operands"]
                ]
                operations.append(node)
            nodes, selected = {}, []
            continue
        match = INSTRUCTION.match(line)
        if match is None:
            continue
        name, shape, opcode, operands, attributes = match.groups()
        counts["instructions"] += 1
        counts["opcode/" + opcode] += 1
        if opcode == "convert" and shape.startswith("bf16["):
            counts["converts_to_bf16"] += 1
        call = CALLS.search(attributes)
        node = dict(name=name, shape=shape, opcode=opcode, calls=call[1] if call else None)
        nodes[name] = node
        metadata = OP_NAME.search(attributes)
        if metadata is None:
            continue
        op_name = metadata[1]
        polar = op_name.endswith("/gram_direction/vmap()/dot_general") or op_name.endswith(
            "jit(polar)/vmap()/dot_general"
        )
        refresh = op_name.endswith("/update_gram_state/dot_general")
        if (polar or refresh) and opcode in ("dot", "convolution", "all-reduce", "reduce-scatter", "all-gather"):
            frame = STACK_FRAME.search(attributes)
            selected.append(
                dict(
                    **node,
                    line=number,
                    computation=scope,
                    family="polar" if polar else "refresh",
                    op_name=op_name,
                    stack_frame_id=int(frame[1]) if frame else None,
                    operands=OPERAND.findall(operands),
                )
            )
    assert not nodes, "Incomplete compiler text"
    return dict(lines=total_lines, static_counts=dict(counts), selected_operations=operations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    directory = StoragePath(args.output_prefix) / ("ir-inventory-" + uuid.uuid4().hex[:12])
    directory.mkdirs()
    output = directory / "report.json"
    report = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        manifest=manifest,
        completed=False,
        artifacts={},
        limitations="Static text inventory, not runtime counts or causal proof. No numerical kernels or training.",
    )
    with tempfile.TemporaryDirectory() as directory:
        for name, artifact in manifest["artifacts"].items():
            compressed = Path(directory) / (name + ".txt.gz")
            digest = hashlib.sha256()
            size = 0
            with StoragePath(artifact["uri"]).open("rb", requests_timeout=60) as source, compressed.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
            assert size == artifact["compressed_bytes"] and digest.hexdigest() == artifact["sha256"], name
            with gzip.open(compressed, "rt") as lines:
                report["artifacts"][name] = inventory(lines)
            report["artifacts"][name]["sha256_verified"] = True
            output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            print(
                json.dumps(
                    dict(artifact=name, selected_operations=len(report["artifacts"][name]["selected_operations"]))
                ),
                flush=True,
            )
            compressed.unlink()
    report["completed"] = True
    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
    print(json.dumps(dict(completed=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
