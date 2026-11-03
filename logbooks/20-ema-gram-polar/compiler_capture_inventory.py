# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract output-mapped records from saved full-update capture compiler graphs."""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from compiler_operation_inventory import (
    CALLS,
    COMPUTATION,
    INSTRUCTION,
    OP_NAME,
    OPERAND,
    inventory,
    polar_lineage,
    producer_bodies,
)
from rigging.filesystem.storage_path import StoragePath


def normalization_records(lines):
    records, callers, scope = [], [], None
    for line in lines:
        header = COMPUTATION.match(line.rstrip())
        if header:
            scope = header[1]
        match = INSTRUCTION.match(line)
        if not match:
            continue
        name, shape, opcode, operands, attributes = match.groups()
        metadata, call = OP_NAME.search(attributes), CALLS.search(attributes)
        if opcode == "fusion" and call:
            callers.append(dict(computation=scope, name=name, calls=call[1], instruction=line.rstrip()))
        if metadata and metadata[1].endswith("/gram_direction/vmap()/div") and opcode == "divide":
            records.append(
                dict(
                    computation=scope,
                    name=name,
                    shape=shape,
                    operands=OPERAND.findall(operands),
                    instruction=line.rstrip(),
                )
            )
    scopes = {row["computation"] for row in records}
    return dict(records=records, callers=[row for row in callers if row["calls"] in scopes])


def output_roles(groups, expected_outputs):
    assert len(groups) == 7 and len(set(groups)) == 7
    assert expected_outputs in (28, 30, 34)
    roles = [f"direction/{group}" for group in groups] + [
        f"state/{group}/{field}" for group in groups for field in ("momentum", "left", "right")
    ]
    captured = groups[:1] if expected_outputs == 30 else groups[:3] if expected_outputs == 34 else []
    roles += [f"capture/{group}/{field}" for group in captured for field in ("inner", "norm_plus_eps")]
    assert len(roles) == expected_outputs
    return roles


def extract(path, groups, expected_outputs):
    with gzip.open(path, "rt") as lines:
        result = inventory(lines)
    diagnostics = result["lineage_diagnostics"] = {}
    with gzip.open(path, "rt") as lines:
        result["polar_lineage"] = polar_lineage(lines, groups, diagnostics)
    outputs = diagnostics["output_polar_ancestors"]
    assert len(outputs) == expected_outputs, (len(outputs), expected_outputs)
    assert all(not ancestors for ancestors in outputs[7:]), "State/capture outputs have polar ancestry"
    result["output_roles"] = output_roles(groups, expected_outputs)
    with gzip.open(path, "rt") as lines:
        result["normalization"] = normalization_records(lines)
    scopes = {row["computation"] for row in result["normalization"]["records"]}
    with gzip.open(path, "rt") as lines:
        bodies = producer_bodies(lines, result["selected_operations"], scopes)
    while True:
        pending = {
            match[1] for body in bodies.values() for line in body for match in CALLS.finditer(line)
        } - bodies.keys()
        if not pending:
            break
        with gzip.open(path, "rt") as lines:
            bodies.update(producer_bodies(lines, [], pending))
    result["producer_bodies"] = bodies
    result["normalization"]["first_product_scope_matches"] = {
        group: [
            row["computation"] + "/" + row["name"]
            for row in result["normalization"]["records"]
            if row["computation"] == products[0]["operation"].split("/", 1)[0]
        ]
        for group, products in result["polar_lineage"]["output_groups"].items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    destination = StoragePath(args.output_prefix) / ("capture-ir-" + uuid.uuid4().hex[:12])
    destination.mkdirs()
    output = destination / "report.json"
    report = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        manifest=manifest,
        completed=False,
        artifacts={},
        limitations=(
            "Static output dependencies, declarations and configurations; not runtime values, precision or cause. "
            "Normalization scope matches are syntactic associations, not proof of consumer identity. "
            "Missing scope matches remain explicit. No optimizer kernels or training."
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        for name, artifact in manifest["artifacts"].items():
            path = Path(directory) / (name + ".txt.gz")
            digest = hashlib.sha256()
            size = 0
            with StoragePath(artifact["uri"]).open("rb", requests_timeout=60) as source, path.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
            assert size == artifact["compressed_bytes"] and digest.hexdigest() == artifact["sha256"], name
            report["active_stage"] = name
            output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            result = extract(path, manifest["groups"], artifact["expected_output_leaves"])
            result["sha256_verified"] = True
            report["artifacts"][name] = result
            output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            print(
                json.dumps(dict(artifact=name, mapped_products=result["polar_lineage"]["assigned_operations"])),
                flush=True,
            )
            path.unlink()
    report["completed"] = True
    report["active_stage"] = "completed"
    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
    print(json.dumps(dict(completed=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
