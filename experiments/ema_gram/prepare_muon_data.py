# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover the pinned public FineWeb cache and verify its experiment-specific copy."""

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.cloud import storage
from rigging.filesystem.transfer import copy


def file_hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    destination = args.destination.rstrip("/")
    bucket_name, prefix = destination.removeprefix("gs://").split("/", 1)
    if bucket_name != "marin-us-central1" or not prefix.startswith("ema-gram-20-geometry/20261113/data/"):
        raise ValueError("Recovery must use the experiment-specific us-central1 data prefix")
    source = f"hf://datasets/{inventory['repo']}@{inventory['revision']}"

    def recover(record):
        path = args.local_root / record["path"]
        expected = record["hf_lfs_sha256"] or record["sha256"]
        hashes = file_hashes(path) if path.exists() and path.stat().st_size == record["bytes"] else None
        if hashes is None or hashes[0] != expected:
            path.parent.mkdir(parents=True, exist_ok=True)
            copy(source + "/" + record["path"], str(path))
            hashes = file_hashes(path)
        sha, md5 = hashes
        if path.stat().st_size != record["bytes"] or sha != expected:
            raise ValueError("Pinned Hugging Face content mismatch: " + record["path"])
        if record["source"] == "gcs" and md5 != record["md5"]:
            raise ValueError("Surviving original dependency differs from pinned public data: " + record["path"])
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(prefix + "/" + record["path"])
        if blob.exists():
            blob.reload()
            if blob.md5_hash != md5 or blob.size != record["bytes"]:
                raise ValueError("Existing experiment copy has different content: " + record["path"])
        else:
            copy(str(path), destination + "/" + record["path"])
            blob.reload()
        if blob.md5_hash != md5 or blob.size != record["bytes"]:
            raise ValueError("Uploaded data checksum mismatch: " + record["path"])
        result = dict(
            path=record["path"],
            bytes=blob.size,
            sha256=sha,
            md5=md5,
            generation=int(blob.generation),
            crc32c=blob.crc32c,
            matches_surviving_dependency=record["source"] == "gcs",
        )
        print(json.dumps(dict(event="file_verified", **result)), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(recover, inventory["records"]))
    records.sort(key=lambda item: item["path"])
    manifest = dict(
        cache_root=destination,
        hf_repo=inventory["repo"],
        hf_revision=inventory["revision"],
        inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        files=records,
        scope=(
            "Every recovered file matches pinned public-source SHA256 and uploaded GCS MD5. "
            "Surviving original dependency objects also match MD5. The deleted training subcache "
            "cannot be compared byte-for-byte; this is an explicit measurement data replacement."
        ),
    )
    content = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    blob = storage.Client().bucket(bucket_name).blob(prefix + "/recovery-manifest.json")
    if blob.exists():
        if blob.download_as_bytes() != content:
            raise ValueError("Published recovery manifest changed")
    else:
        blob.upload_from_string(content, content_type="application/json", if_generation_match=0)
    print(
        json.dumps(
            dict(
                event="cache_verified",
                manifest_uri=destination + "/recovery-manifest.json",
                manifest_sha256=hashlib.sha256(content).hexdigest(),
                files=len(records),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
