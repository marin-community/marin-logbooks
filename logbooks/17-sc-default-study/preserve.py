# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive stopped study scratch with full content hashes and exact restore paths."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from rigging.filesystem.buckets import filesystem_for

SOURCE_BASE = "s3://marin-rno2a/tmp/ttl=14d/skyrl/marin/users/romain/checkpoints/sc-default-study/"
ARCHIVE_BASES = (
    "s3://marin-rno2a/marin/users/romain/research/sc-default-study-01a16b97-e977-75f3-8b2b-fd07bbd1c609/",
    "s3://marin-us-east-02a/marin/users/romain/research/sc-default-study-01a16b97-e977-75f3-8b2b-fd07bbd1c609/",
)
BLOCK_BYTES = 64 * 2**20


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read(uri):
    fs, key = filesystem_for(uri)
    return fs.cat_file(key)


def write(uri, payload):
    fs, key = filesystem_for(uri)
    fs.pipe_file(key, payload)


def immutable(uri, payload):
    fs, key = filesystem_for(uri)
    if fs.exists(key):
        if fs.cat_file(key) != payload:
            raise ValueError(f"Existing immutable object differs: {uri}")
    else:
        fs.pipe_file(key, payload)
    if fs.cat_file(key) != payload:
        raise ValueError(f"Immutable object readback differs: {uri}")


def source_members(fs, prefix):
    fs.invalidate_cache()
    return {
        key: (row["size"], row["ETag"], str(row["LastModified"]))
        for key, row in fs.find(prefix, detail=True).items()
        if row["type"] == "file"
    }


def check_prefixes(source, archive):
    if not source.startswith(SOURCE_BASE) or not source.endswith("/") or source == SOURCE_BASE:
        raise ValueError("Source must be one stopped run under the owned study scratch")
    parts = source.removeprefix(SOURCE_BASE).strip("/").split("/")
    valid_tree = len(parts) == 2 or (len(parts) == 3 and parts[-1] == "checkpoints")
    if not valid_tree or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Source must identify one run/version or its checkpoint tree")
    if not any(archive.startswith(base) and archive != base for base in ARCHIVE_BASES) or archive.endswith("/"):
        raise ValueError("Archive must be a named member of this session's owned namespace")


def inventory(source, archive, run_source_revision):
    check_prefixes(source, archive)
    fs, prefix = filesystem_for(source)
    members = source_members(fs, prefix)
    if not members:
        raise ValueError("Source inventory is empty")
    files = []
    for key, (size, etag, modified) in sorted(members.items()):
        identity = {"source": "s3://" + key, "etag": etag, "bytes": size}
        # Match the first archive trial's object names so its copy can be verified and reused.
        revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        files.append(
            {
                **identity,
                "last_modified": modified,
                "archive": archive + "/objects/" + revision + "/" + PurePosixPath(key).name,
            }
        )
    manifest = {
        "schema": "sc-default-run-preservation-input/3",
        "source_prefix": source,
        "run_source_revision": run_source_revision,
        "archive_prefix": archive,
        "files": files,
        "source_deletions": 0,
    }
    raw = canonical(manifest)
    uri = archive + "/manifests/" + hashlib.sha256(raw).hexdigest() + ".json"
    immutable(uri, raw)
    return {"manifest_uri": uri, "objects": len(files), "bytes": sum(row["bytes"] for row in files)}


def checksum(fs, key, size):
    digest, count = hashlib.sha256(), 0
    with fs.open(key, "rb", block_size=BLOCK_BYTES, cache_type="none") as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
            count += len(block)
    if count != size:
        raise ValueError(f"Unexpected object length: {key}")
    return digest.hexdigest()


def stream_copy(source_fs, source_key, archive_fs, archive_key, size):
    digest, count = hashlib.sha256(), 0
    with source_fs.open(source_key, "rb", block_size=BLOCK_BYTES, cache_type="none") as src:
        with archive_fs.open(archive_key, "wb", block_size=BLOCK_BYTES) as dst:
            while block := src.read(BLOCK_BYTES):
                digest.update(block)
                count += len(block)
                dst.write(block)
    if count != size:
        raise ValueError(f"Unexpected streamed source length: {source_key}")
    return digest.hexdigest()


def preserve(manifest_uri, execution_source_revision):
    raw = read(manifest_uri)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != PurePosixPath(manifest_uri).stem:
        raise ValueError("Input manifest digest differs from its immutable name")
    manifest = json.loads(raw)
    source, archive = manifest["source_prefix"], manifest["archive_prefix"]
    check_prefixes(source, archive)
    rows = manifest["files"]
    if not rows or len({row["source"] for row in rows}) != len(rows):
        raise ValueError("Input has no members or repeats a source object")
    if any(not row["source"].startswith(source) or not row["archive"].startswith(archive + "/objects/") for row in rows):
        raise ValueError("Object escaped the input's source/archive prefixes")
    fs, prefix = filesystem_for(source)
    archive_fs, _ = filesystem_for(archive)
    same_bucket = source.split("/")[2] == archive.split("/")[2]
    expected = {row["source"].removeprefix("s3://"): (row["bytes"], row["etag"], row["last_modified"]) for row in rows}

    def unchanged():
        if source_members(fs, prefix) != expected:
            raise ValueError("Source tree changed; a fresh inventory is required")

    unchanged()
    checkpoint_uri = archive + "/verified-members-v3.json"
    checkpoint_fs, checkpoint_key = filesystem_for(checkpoint_uri)
    checkpoint = json.loads(read(checkpoint_uri)) if checkpoint_fs.exists(checkpoint_key) else {}
    if checkpoint and checkpoint["input_manifest_sha256"] != manifest_sha:
        raise ValueError("Existing verification belongs to a different input inventory")
    verified = checkpoint.get("members", {})

    def copy(row):
        source_key = row["source"].removeprefix("s3://")
        archive_key = row["archive"].removeprefix("s3://")
        previous = verified.get(row["source"])
        try:
            observed = archive_fs.info(archive_key)
        except FileNotFoundError:
            observed = None
        if previous and observed and previous["archive_etag"] == observed["ETag"] and observed["size"] == row["bytes"]:
            return previous
        copy_method = "existing unverified copy"
        if observed is None and not same_bucket:
            source_sha = stream_copy(fs, source_key, archive_fs, archive_key, row["bytes"])
            copy_method = "streamed between buckets"
            archive_fs.invalidate_cache(archive_key)
            observed = archive_fs.info(archive_key)
        else:
            source_sha = checksum(fs, source_key, row["bytes"])
            if observed is None:
                fs.cp_file(source_key, archive_key)
                copy_method = "same-bucket server copy"
                archive_fs.invalidate_cache(archive_key)
                observed = archive_fs.info(archive_key)
        archive_sha = checksum(archive_fs, archive_key, row["bytes"]) if observed["size"] == row["bytes"] else None
        streamed_recovery = False
        if source_sha != archive_sha:
            # Only this unverified derived archive member is replaced; source stays read-only.
            if stream_copy(fs, source_key, archive_fs, archive_key, row["bytes"]) != source_sha:
                raise ValueError(f"Source bytes changed during repair: {row['source']}")
            archive_sha = checksum(archive_fs, archive_key, row["bytes"])
            streamed_recovery = True
        if source_sha != archive_sha:
            raise ValueError(f"Archive bytes differ from source: {row['source']}")
        archive_fs.invalidate_cache(archive_key)
        observed = archive_fs.info(archive_key)
        return {
            **row,
            "sha256": source_sha,
            "archive_etag": observed["ETag"],
            "copy_method": copy_method,
            "streamed_recovery": streamed_recovery,
        }

    def save(status):
        write(checkpoint_uri, canonical({"input_manifest_sha256": manifest_sha, "members": verified}))
        progress = {
            "status": status,
            "verified_objects": len(verified),
            "total_objects": len(rows),
            "verified_bytes": sum(row["bytes"] for row in verified.values()),
            "total_bytes": sum(row["bytes"] for row in rows),
            "source_deletions": 0,
        }
        write(archive + "/progress-v3.json", canonical(progress))
        print(json.dumps(progress), flush=True)

    save("copying and verifying with two CPU threads")
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        pending = [pool.submit(copy, row) for row in rows]
        for future in as_completed(pending):
            result = future.result()
            verified[result["source"]] = result
            if len(verified) % 250 == 0 or result["bytes"] >= 2**30:
                save("copying and verifying with two CPU threads")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        save("partial verification retained")
    if set(verified) != {row["source"] for row in rows}:
        raise ValueError("Missing verified archive members")
    unchanged()
    complete = {
        "schema": "sc-default-run-preservation/3",
        "input_manifest_uri": manifest_uri,
        "input_manifest_sha256": manifest_sha,
        "execution_source_revision": execution_source_revision,
        "run_source_revision": manifest["run_source_revision"],
        "tool_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_prefix": source,
        "archive_prefix": archive,
        "files": [verified[row["source"]] for row in rows],
        "all_source_archive_sha256_equal": True,
        "source_tree_unchanged": True,
        "source_deletions": 0,
        "restore": "Restore each archive member to its exact source URI; verify its recorded SHA256 and length.",
    }
    body = canonical(complete)
    digest = hashlib.sha256(body).hexdigest()
    complete_uri = archive + "/complete-manifests/" + digest + ".json"
    immutable(complete_uri, body)
    proof = {
        "manifest_uri": complete_uri,
        "manifest_sha256": digest,
        "objects": len(verified),
        "bytes": sum(row["bytes"] for row in rows),
        "all_source_archive_sha256_equal": True,
        "source_tree_unchanged": True,
        "source_deletions": 0,
        "restore": complete["restore"],
    }
    write(archive + "/proof-v3.json", canonical(proof))
    save("complete")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("inventory")
    listing.add_argument("source")
    listing.add_argument("archive")
    listing.add_argument("--run-source-revision", required=True)
    copying = commands.add_parser("copy")
    copying.add_argument("manifest_uri")
    copying.add_argument("--execution-source-revision", required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.source, args.archive, args.run_source_revision)
    else:
        result = preserve(args.manifest_uri, args.execution_source_revision)
    print(json.dumps(result), flush=True)
