# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory saved substitution graphs on regional CPU, with source-labelled inputs."""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from compiler_capture_inventory import extract
from compiler_input_provenance import provenance
from compiler_normalization_closure import complete_graph
from rigging.filesystem.storage_path import StoragePath


def external_bindings(parameters, expectations):
    """Bind source pytree labels, never surviving compiled parameter ordinals."""
    result = {}
    for role, expected in expectations.items():
        matches = [(index, row) for index, row in parameters.items() if row["source_label"] == expected["source_label"]]
        assert len(matches) == 1, (role, matches)
        index, row = matches[0]
        assert row["shape"].split("{", 1)[0] == expected["shape"], (role, row)
        result[role] = dict(compiled_parameter_index=int(index), **row, source_argument=expected["source_argument"])
    assert len({row["compiled_parameter_index"] for row in result.values()}) == len(result)
    return result


def enrich(path, graph, name, artifact):
    closure = complete_graph(name, graph, artifact, path)
    targets = {
        group + "/first_product": dict(operation=chain[0]["operation"], role="first polar Gram product")
        for group, chain in graph["polar_lineage"]["output_groups"].items()
    }
    divisions = {}
    for group, operands in closure["trace"].items():
        found = {
            row["computation"] + "/" + row["name"] for operand in operands for row in operand["normalization_divisions"]
        }
        divisions[group] = sorted(found)
    # Some other groups stop at opaque calls. The q target must be unambiguous.
    assert len(divisions["q_proj"]) == 1, divisions["q_proj"]
    targets["q_proj/normalization"] = dict(operation=divisions["q_proj"][0], role="q normalization division")
    with gzip.open(path, "rt") as lines:
        inputs = provenance(lines, targets)
    bindings = external_bindings(inputs["entry_parameters"], artifact.get("external_inputs", {}))
    if bindings:
        division = inputs["targets"]["q_proj/normalization"]["operand_provenance"]
        for index, role in enumerate(("inner", "norm_plus_eps")):
            row = division[index]
            assert not row["conservative_operations"], row
            assert len(row["entry_parameters"]) == 1, row
            param = row["entry_parameters"][0]
            assert param["index"] == bindings[role]["compiled_parameter_index"] and param["tuple_path"] == [], row
        external = {row["compiled_parameter_index"] for row in bindings.values()}
        for group in graph["polar_lineage"]["output_groups"]:
            if group != "q_proj":
                for operand in inputs["targets"][group + "/first_product"]["operand_provenance"]:
                    assert not external.intersection(p["index"] for p in operand["entry_parameters"]), group
    graph["normalization_closure"] = closure
    graph["normalization_by_group"] = divisions
    graph["input_provenance"] = inputs
    graph["external_input_bindings"] = bindings
    graph["first_gram_collectives"] = {
        group: [
            dict(operand_index=i, **row)
            for row in graph["lineage_diagnostics"]["entry_collective_inputs"]
            for i, ancestors in enumerate(row["polar_ancestors_by_operand"])
            if ancestors == [chain[0]["operation"]]
        ]
        for group, chain in graph["polar_lineage"]["output_groups"].items()
    }
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    directory = StoragePath(args.output_prefix) / ("substitution-ir-" + uuid.uuid4().hex[:12])
    directory.mkdirs()
    output = directory / "report.json"
    report = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        completed=False,
        artifacts={},
        source_sha256={
            p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in manifest["source_files"]
        },
        limitations=(
            "Static declared dependencies/configurations only; opaque boundaries retained. "
            "Both numerical substitution gates failed. No runtime values, precision, order, "
            "internal identity or causal conclusion."
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        for name, artifact in manifest["artifacts"].items():
            path = Path(temporary) / "graph.gz"
            digest, size = hashlib.sha256(), 0
            with StoragePath(artifact["uri"]).open("rb", requests_timeout=60) as source, path.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
            assert digest.hexdigest() == artifact["sha256"] and size == artifact["compressed_bytes"], name
            graph = extract(path, manifest["groups"], artifact["expected_output_leaves"])
            graph["sha256_verified"] = True
            report["artifacts"][name] = enrich(path, graph, name, artifact)
            report["active_stage"] = name
            output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            print(
                json.dumps(dict(graph=name, products=graph["polar_lineage"]["assigned_operations"], output=str(output))),
                flush=True,
            )
            path.unlink()
    report["completed"] = True
    report["active_stage"] = "completed"
    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
    print(json.dumps(dict(completed=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
