# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract explicitly named computation bodies from checksum-verified regional HLO."""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from compiler_operation_inventory import CALLS, producer_bodies
from rigging.filesystem.storage_path import StoragePath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    directory = StoragePath(args.output_prefix) / ("selected-bodies-" + uuid.uuid4().hex[:12])
    directory.mkdirs()
    output = directory / "report.json"
    report = dict(
        source_commit=os.environ.get("GIT_COMMIT"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        manifest=manifest,
        artifacts={},
        completed=False,
        limitations=(
            "Selected static bodies; backend configs omitted from body lines. " "No kernels or runtime-value claim."
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        for name, artifact in manifest["artifacts"].items():
            path = Path(temporary) / (name + ".txt.gz")
            digest, size = hashlib.sha256(), 0
            with StoragePath(artifact["uri"]).open("rb", requests_timeout=60) as source, path.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
            assert size == artifact["compressed_bytes"] and digest.hexdigest() == artifact["sha256"], name
            bodies, pending = {}, set(artifact["computation_names"])
            while pending:
                with gzip.open(path, "rt") as lines:
                    captured = producer_bodies(lines, [], pending)
                bodies.update(captured)
                pending = {
                    match[1] for body in captured.values() for line in body for match in CALLS.finditer(line)
                } - bodies.keys()
            report["artifacts"][name] = dict(sha256_verified=True, input_sha256=digest.hexdigest(), bodies=bodies)
            report["active_stage"] = name
            output.write_text(json.dumps(report, indent=2), requests_timeout=60)
            print(json.dumps(dict(artifact=name, bodies=len(bodies))), flush=True)
            path.unlink()
    report["completed"] = True
    report["active_stage"] = "completed"
    output.write_text(json.dumps(report, indent=2), requests_timeout=60)
    print(json.dumps(dict(completed=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
