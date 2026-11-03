# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover only missing producers of the dependency-selected earlier product and collective inputs."""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from compiler_operation_inventory import producer_bodies
from compiler_polar_boundaries import missing_scopes, verified_json
from rigging.filesystem.storage_path import StoragePath
from trace_polar_initializers import analyze, select_targets


def complete_graph(graph, reports, targets, path):
    graph = dict(graph, producer_bodies=dict(graph["producer_bodies"]))
    rounds = []
    while True:
        result = analyze(graph, reports, targets)
        missing = missing_scopes(result)
        if not missing:
            return dict(trace=result, extraction_rounds=rounds)
        assert not missing.intersection(graph["producer_bodies"]), "Known body still unresolved"
        with gzip.open(path, "rt") as lines:
            bodies = producer_bodies(lines, [], missing)
        assert missing == bodies.keys()
        graph["producer_bodies"].update(bodies)
        rounds.append(sorted(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    root = Path(__file__).parent
    source_sha = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in manifest["source_files"]}
    assert source_sha == manifest["source_files"]
    source = verified_json(manifest["inventory"])
    reports = {name: verified_json(entry) for name, entry in manifest["reports"].items()}
    assert all(r["completed"] for r in (source, *reports.values()))
    output = StoragePath(args.output_prefix) / ("polar-initializers-" + uuid.uuid4().hex[:12]) / "report.json"
    output.parent.mkdirs()
    report = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        source_sha256=source_sha,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        manifest=manifest,
        artifacts={},
        completed=False,
        limitations="Static declared bindings only; both numerical full28leaf substitution gates remain failed.",
    )
    with tempfile.TemporaryDirectory() as directory:
        for name in manifest["graphs"]:
            graph = source["artifacts"][name]
            artifact = source["manifest"]["artifacts"][name]
            assert graph["sha256_verified"]
            assert all(r["artifacts"][name]["input_sha256"] == artifact["sha256"] for r in reports.values())
            saved = {label: r["artifacts"][name]["trace"] for label, r in reports.items()}
            targets = select_targets(graph, saved["state-initializer"], saved)
            assert targets == manifest["targets"][name]
            path = Path(directory) / "graph.txt.gz"
            digest, size = hashlib.sha256(), 0
            with StoragePath(artifact["uri"]).open("rb", requests_timeout=60) as reader, path.open("wb") as writer:
                while chunk := reader.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    writer.write(chunk)
            assert size == artifact["compressed_bytes"] and digest.hexdigest() == artifact["sha256"]
            result = complete_graph(graph, saved, targets, path)
            report["artifacts"][name] = dict(input_sha256=digest.hexdigest(), sha256_verified=True, **result)
            output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            print(json.dumps(dict(graph=name, extraction_rounds=result["extraction_rounds"])), flush=True)
            path.unlink()
    report["completed"] = True
    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
    print(json.dumps(dict(completed=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
