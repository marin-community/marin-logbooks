# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist available files with no application retry loop; SDK defaults remain unchanged."""

import hashlib
import json
import tarfile
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def upload_new(path, uri):
    fs, key = filesystem_for(uri)
    if fs.exists(key):
        raise FileExistsError("Refusing to overwrite an execution artifact")
    fs.put_file(str(path), key)
    observed = hashlib.sha256()
    size = 0
    with fs.open(key, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            observed.update(chunk)
    if size != Path(path).stat().st_size or observed.hexdigest() != digest(path):
        raise ValueError("Execution artifact remote readback differs")


def persist_tree(output, root, unit):
    if not root.startswith("s3://marin-rno2a/tmp/ttl=14d/"):
        raise ValueError("Execution persistence requires RNO14d outputs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    records = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    archive = output.parent / f"{unit}-available-output.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for path in files:
            bundle.add(path, arcname=str(path.relative_to(output)), recursive=False)
    archive_uri = f"{root}/{unit}/available-output.tar.gz"
    upload_new(archive, archive_uri)
    # Mirror files too, preserving the scientific reader's expected paths.
    for path in files:
        upload_new(path, f"{root}/{path.relative_to(output).as_posix()}")
    receipt = {
        "archive": archive_uri,
        "archive_sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "files": records,
        "coverage": "available files only; aggregate callback provides no per-response crash persistence",
        "verification": "full remote SHA256 and byte length for every uploaded object",
        "application_retries": 0,
        "transport_retry_policy": "s3fs/SDK defaults unchanged; not asserted single-attempt",
    }
    receipt_path = output.parent / f"{unit}-persistence.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    upload_new(receipt_path, f"{root}/{unit}/persistence.json")
    return receipt
