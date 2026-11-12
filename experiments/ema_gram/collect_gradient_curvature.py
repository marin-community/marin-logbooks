# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect small probe records while preserving separate process attempts."""

import argparse
import hashlib
import json
from pathlib import Path

from google.api_core.exceptions import NotFound
from google.cloud import storage

RECORD_NAMES = {
    "configuration.json",
    "data-manifest.json",
    "path-results.json",
    "adaptive-results.json",
    "adaptive-plan.json",
    "measurements.jsonl",
    "completed.json",
    "failure.json",
}


def current_record(client, bucket, name):
    """Refresh generation on overwrite races, without replacing a good local copy."""
    for _ in range(3):
        blob = client.bucket(bucket).get_blob(name)
        if blob is None:
            continue
        if blob.size > 20_000_000:
            raise ValueError("Unexpected artifact size")
        try:
            return blob, blob.download_as_bytes()
        except NotFound:
            # Live probes atomically overwrite these small objects. The listed
            # generation may disappear between listing and downloading it.
            continue
    raise RuntimeError(f"Artifact changed during three reads: gs://{bucket}/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bucket, prefix = args.prefix.removeprefix("gs://").rstrip("/").split("/", 1)
    args.output.mkdir(parents=True, exist_ok=True)
    records, attempts = [], {}
    client = storage.Client()
    for blob in client.list_blobs(bucket, prefix=prefix + "/"):
        relative = Path(blob.name.removeprefix(prefix + "/"))
        if relative.name not in RECORD_NAMES:
            continue
        if relative.is_absolute() or ".." in relative.parts or blob.size > 20_000_000:
            raise ValueError("Unexpected artifact path or size")
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob, content = current_record(client, bucket, blob.name)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        records.append(
            dict(
                source=f"gs://{bucket}/{blob.name}",
                path=str(relative),
                generation=blob.generation,
                bytes=blob.size,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            )
        )
        attempt = attempts.setdefault(str(relative.parent), dict(files=[]))
        attempt["files"].append(relative.name)
        if relative.name in ("completed.json", "failure.json"):
            attempt[relative.stem] = json.loads(destination.read_text())
    index = dict(
        prefix=args.prefix,
        records=records,
        attempts=attempts,
        rule="Attempts are separate; controller success is additionally required for completion",
    )
    (args.output / "collection-index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(records=len(records), attempts=attempts), indent=2))


if __name__ == "__main__":
    main()
