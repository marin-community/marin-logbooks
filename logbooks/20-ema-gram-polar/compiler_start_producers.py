# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover declared inputs of opaque endpoints without assigning custom-call semantics."""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from compiler_operation_inventory import producer_bodies
from rigging.filesystem.storage_path import StoragePath
from trace_start_producers import analyze


def missing_scopes(value):
    if isinstance(value, dict):
        found = {name.split("/", 1)[0] for name in value.get("missing_records", [])}
        for child in value.values():
            found.update(missing_scopes(child))
        return found
    if isinstance(value, list):
        return set().union(*(missing_scopes(child) for child in value))
    return set()


def complete_graph(graph, outer, boundary, path):
    """Read only bodies requested by the binding tracer until no record is missing."""
    graph = dict(graph, producer_bodies=dict(graph["producer_bodies"]))
    rounds = []
    while True:
        result = analyze(graph, outer, boundary)
        missing = missing_scopes(result)
        if not missing:
            return dict(trace=result, extraction_rounds=rounds)
        assert not missing.intersection(graph["producer_bodies"]), "Known body still unresolved"
        with gzip.open(path, "rt") as lines:
            bodies = producer_bodies(lines, [], missing)
        assert missing == bodies.keys()
        graph["producer_bodies"].update(bodies)
        rounds.append(sorted(missing))


def verified_json(artifact):
    raw = StoragePath(artifact["uri"]).read_bytes(requests_timeout=60)
    assert len(raw) == artifact["bytes"] and hashlib.sha256(raw).hexdigest() == artifact["sha256"]
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    source = verified_json(manifest["inventory"])
    outer = verified_json(manifest["outer_report"])
    boundary = verified_json(manifest["boundary_report"])
    assert source["completed"] and outer["completed"]
    directory = StoragePath(args.output_prefix) / ("start-producers-" + uuid.uuid4().hex[:12])
    directory.mkdirs()
    output = directory / "report.json"
    report = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tracer_sha256=hashlib.sha256(Path(__file__).with_name("trace_start_producers.py").read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        manifest=manifest,
        artifacts={},
        completed=False,
        limitations=(
            "Static dependency coverage only; opaque boundaries retained. "
            "Both numerical substitution full-reference gates failed."
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        for name in manifest["graphs"]:
            original = source["artifacts"][name]
            artifact = source["manifest"]["artifacts"][name]
            graph = dict(original, producer_bodies=dict(original["producer_bodies"]))
            path = Path(temporary) / "graph.txt.gz"
            digest, size = hashlib.sha256(), 0
            with StoragePath(artifact["uri"]).open("rb", requests_timeout=60) as reader, path.open("wb") as writer:
                while chunk := reader.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    writer.write(chunk)
            assert size == artifact["compressed_bytes"] and digest.hexdigest() == artifact["sha256"]
            assert outer["artifacts"][name]["input_sha256"] == artifact["sha256"]
            result = complete_graph(graph, outer["artifacts"][name]["trace"], boundary["artifacts"][name]["trace"], path)
            report["artifacts"][name] = dict(input_sha256=digest.hexdigest(), sha256_verified=True, **result)
            output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            print(json.dumps(dict(graph=name, extraction_rounds=result["extraction_rounds"])), flush=True)
            path.unlink()
    report["completed"] = True
    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
    print(json.dumps(dict(completed=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
