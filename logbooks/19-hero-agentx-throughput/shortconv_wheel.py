# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repack the baseline wheel with pinned Grug Python source replacements.

This creates a private experiment input, never an official release. All packaged
files must match the installed wheel's RECORD before replacement.
Dependencies, compiled extensions, metadata and the version stay fixed.
"""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import zipfile
from pathlib import Path

MODEL_FILE = "vllm/model_executor/models/grugmoe.py"
BASE_MODEL_SHA256 = "9f6404a1a6c25046a675fe4439cc4295e1a6f34ec71ae0614c58a0bb3275e8e3"
BASE_WHEEL_SHA256 = "e8520d4493e5a60f9f6aa96ea3bc80c42e45258e4d2c45882869fc5921df3e15"


def encoded_digest(digest: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--replacement-source", type=Path, required=True)
    parser.add_argument("--replacement-sha256", required=True)
    parser.add_argument("--config-source", type=Path)
    parser.add_argument("--config-source-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replacement = args.replacement_source.read_bytes()
    assert hashlib.sha256(replacement).hexdigest() == args.replacement_sha256
    replacements = {MODEL_FILE: replacement}
    if args.config_source is not None:
        config_source = args.config_source.read_bytes()
        assert hashlib.sha256(config_source).hexdigest() == args.config_source_sha256
        replacements["vllm/model_executor/models/config.py"] = config_source
    distribution = importlib.metadata.distribution("vllm")
    assert hashlib.sha256(Path(distribution.locate_file(MODEL_FILE)).read_bytes()).hexdigest() == BASE_MODEL_SHA256
    record = next(path for path in distribution.files if path.name == "RECORD")
    entries = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for item in sorted(distribution.files):
            if (
                ".." in item.parts
                or "__pycache__" in item.parts
                or item.name in ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json")
            ):
                continue
            path = Path(distribution.locate_file(item))
            with path.open("rb") as stream:
                original_digest = hashlib.file_digest(stream, "sha256").digest()
            assert item.hash is not None and item.hash.mode == "sha256", item
            assert encoded_digest(original_digest).split("=", 1)[1] == item.hash.value, item
            if str(item) in replacements:
                content = replacements[str(item)]
                archive.writestr(str(item), content)
                digest, size = hashlib.sha256(content).digest(), len(content)
            else:
                archive.write(path, str(item))
                digest, size = original_digest, path.stat().st_size
            entries.append((str(item), encoded_digest(digest), size))
        entries.append((str(record), "", ""))
        text = io.StringIO(newline="")
        csv.writer(text).writerows(entries)
        archive.writestr(str(record), text.getvalue())
    with args.output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "kind": "private_python_overlay",
        "base_wheel_sha256": BASE_WHEEL_SHA256,
        "source_commit": args.source_commit,
        "changed_files": {name: hashlib.sha256(content).hexdigest() for name, content in replacements.items()},
        "wheel_path": str(args.output.resolve()),
        "wheel_sha256": digest,
        "wheel_bytes": args.output.stat().st_size,
        "verified_unchanged_files": len(entries) - 1 - len(replacements),
        "version": distribution.version,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
