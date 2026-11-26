"""Publish the V10 release's artifact-graph record (sft-mixes/exp3-v10/release@<version>).

The record adopts the frozen publisher manifest: its config carries the manifest
digest and the pinned members, exactly as the V7/V8/V9 release records did. The
long-loop runner re-derives the handle with ArtifactStep.adopt and compares
fingerprints, so this record and the manifest must not drift.
"""

import argparse
import hashlib
import json

from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.s3_compat import configure_coreweave_s3


def main():
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True, help="The frozen publisher manifest URI")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--members", required=True, type=json.loads,
                        help='The manifest members, as {"name": {"uri": ..., "sha256": ...}}')
    parser.add_argument("--name", default="sft-mixes/exp3-v10/release")
    parser.add_argument("--version", required=True)
    parser.add_argument("--graph-prefix", required=True, help="Where the record is written, e.g. .../artifact-graph/sft-mixes/exp3-v10/release/<version>")
    args = parser.parse_args()

    source = args.manifest_uri
    if source.endswith("/manifest.json"):
        source = source[: -len("manifest.json")] + "manifest"  # the adopted source pins the manifest stem, as V7/V8 did
    handle = ArtifactStep.adopt(
        args.name,
        args.version,
        source,
        config={"sha256": args.manifest_sha256, "members": args.members},
    )
    record = ArtifactRecord(
        name=handle.name,
        version=handle.version,
        fingerprint=handle.fingerprint(),
        result_type="marin.execution.artifact.Artifact",
        output_path=args.graph_prefix.rstrip("/"),
        deps=[],
        dep_paths=[],
        config=handle.adopt_config,
        source=handle.adopt_source,
        result=None,
        fingerprint_payload=handle.fingerprint_payload(),
    )
    write_record(record)
    record_uri = f"{args.graph_prefix.rstrip('/')}/.artifact.json"
    print(json.dumps({
        "record_uri": record_uri,
        "fingerprint": record.fingerprint,
        "sha256": hashlib.sha256(
            record.model_dump_json(indent=2).encode()
        ).hexdigest(),
        "payload_matches": record.fingerprint_payload == handle.fingerprint_payload(),
    }))


if __name__ == "__main__":
    main()
