# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect small probe records while preserving separate process attempts."""

import argparse
import hashlib
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bucket, prefix = args.prefix.removeprefix("gs://").rstrip("/").split("/", 1)
    args.output.mkdir(parents=True, exist_ok=True)
    records, attempts = [], {}
    for blob in storage.Client().list_blobs(bucket, prefix=prefix + "/"):
        relative = Path(blob.name.removeprefix(prefix + "/"))
        if relative.name not in RECORD_NAMES:
            continue
        if relative.is_absolute() or ".." in relative.parts or blob.size > 20_000_000:
            raise ValueError("Unexpected artifact path or size")
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(destination)
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
