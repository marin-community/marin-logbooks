# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the RECORD-verified private overlay packager on the current main wheel."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.inputs / "marin-vllm-gpu-manifest.json").read_text())
    platform = next(p for p in manifest["platforms"] if p["architecture"] == "aarch64")
    base = args.inputs / platform["wheel"]["filename"]
    assert digest(base) == platform["wheel"]["sha256"]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=args.source)
    extracted = args.inputs / "official-wheel-files"
    extracted.mkdir(exist_ok=False)
    with zipfile.ZipFile(base) as archive:
        for member in archive.infolist():
            path = Path(archive.extract(member, extracted))
            if mode := member.external_attr >> 16:
                path.chmod(mode & 0o7777)
    output = args.inputs / "private" / base.name
    modules = [
        "vllm/model_executor/models/grugmoe.py",
        "vllm/v1/attention/backends/flashinfer.py",
        "vllm/v1/attention/backends/grug_logical_kv.py",
    ]
    command = [
        sys.executable, str(Path(__file__).with_name("shortconv_wheel.py")),
        "--source-commit", source,
        "--replacement-source", str(args.source / modules[0]),
        "--replacement-sha256", digest(args.source / modules[0]),
        "--base-module-sha256", digest(extracted / modules[0]),
        "--base-wheel-sha256", digest(base),
        "--additional-source", modules[1], str(args.source / modules[1]),
        digest(args.source / modules[1]), digest(extracted / modules[1]),
        "--new-source", modules[2], str(args.source / modules[2]), digest(args.source / modules[2]),
        "--output", str(output),
    ]
    subprocess.run(command, env=dict(os.environ) | {"PYTHONPATH": str(extracted)}, check=True)
    receipt = json.loads(output.with_suffix(".json").read_text())
    assert receipt["source_commit"] == source
    assert (extracted / "vllm/utils/numa_wrapper.sh").stat().st_mode & 0o111
    print(json.dumps({"release_source": manifest["source"], "overlay": receipt}), flush=True)


if __name__ == "__main__":
    main()
