# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable per-job/attempt snapshots; never overwrite differing artifact bytes."""

import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path

from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.buckets import filesystem_for


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def native_execution_id() -> str:
    info = get_job_info()
    if info is None:
        raise RuntimeError("Persistence requires an explicit native job/attempt identity")
    return f"{info.job_id.to_safe_token()}-attempt-{info.attempt_id}"


def upload_immutable(path, uri):
    """Reuse identical objects after full readback; differing bytes remain protected."""
    fs, key = filesystem_for(uri)
    exists = fs.exists(key)
    if not exists:
        fs.put_file(str(path), key)
    observed = hashlib.sha256()
    size = 0
    with fs.open(key, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            observed.update(chunk)
    if size != Path(path).stat().st_size or observed.hexdigest() != digest(path):
        if exists:
            raise FileExistsError(f"Existing artifact has different bytes; preserved without overwrite: {uri}")
        raise ValueError("Execution artifact remote readback differs")


def persist_tree(output, root, unit, execution_id: str, *, mirror_prefixes: tuple[str, ...] = ()):
    if not root.startswith("s3://marin-rno2a/tmp/ttl=14d/"):
        raise ValueError("Execution persistence requires RNO14d outputs")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", execution_id):
        raise ValueError("Execution identity must be a safe native job token plus attempt")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    records = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    archive = output.parent / f"{unit}-{execution_id}-available-output.tar.gz"
    # Stable metadata allows an identical finalization to verify/reuse its bytes.
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for path in files:
                member = tarfile.TarInfo(path.relative_to(output).as_posix())
                member.size = path.stat().st_size
                member.mode = 0o644
                with path.open("rb") as stream:
                    bundle.addfile(member, stream)
    archive_sha = digest(archive)
    scope = f"{root}/{unit}/{execution_id}/snapshots/{archive_sha}"
    archive_uri = f"{scope}/available-output.tar.gz"
    upload_immutable(archive, archive_uri)
    receipt = {
        "execution_id": execution_id,
        "archive": archive_uri,
        "archive_sha256": archive_sha,
        "archive_bytes": archive.stat().st_size,
        "files": records,
        "coverage": "available files only; aggregate callback provides no per-response crash persistence",
        "verification": "full remote SHA256 and byte length; identical existing bytes reused without PUT",
        "application_retries": 0,
        "transport_retry_policy": "s3fs/SDK defaults unchanged; not asserted single-attempt",
    }
    receipt_path = output.parent / f"{unit}-{execution_id}-persistence.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    upload_immutable(receipt_path, f"{scope}/persistence.json")
    # Always publish the recoverable snapshot before any scientific-reader alias.
    print(json.dumps({"persistence_snapshot": scope, "execution_id": execution_id}), flush=True)
    for path in files:
        relative = path.relative_to(output)
        if relative.parts[0] in mirror_prefixes:
            upload_immutable(path, f"{root}/{relative.as_posix()}")
    return receipt
