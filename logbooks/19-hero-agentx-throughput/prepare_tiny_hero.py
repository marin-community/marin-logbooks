# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the existing Hero fixture producer on CPU with the pinned tokenizer.

This fixture qualifies changing-batch graph/state behavior. Throughput runs
continue to use the full step-144000 checkpoint. The reference package comes
from an exact MarinSkyRL archive; no serving dependency is installed or changed.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.manifest.read_text())
    torch.set_num_threads(1)
    namespace = {"__name__": "tiny_hero_fixture", "load_file": load_file, "save_file": save_file}
    for relative, expected in specification["fixture_source_sha256"].items():
        path = args.source_root / relative
        source = path.read_bytes()
        assert hashlib.sha256(source).hexdigest() == expected
        if path.name == "grug_tiny_checkpoint.py":
            exec(compile(source, str(path), "exec"), namespace)
        else:
            tree = ast.parse(source)
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "write_tiny_hero_checkpoint"
            )
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    assert (
        hashlib.sha256((args.tokenizer / "tokenizer.json").read_bytes()).hexdigest() == specification["tokenizer_sha256"]
    )
    namespace["TOKENIZER"] = str(args.tokenizer)
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.device("cpu"):
        namespace["write_tiny_hero_checkpoint"](args.output)
    files = {}
    for path in sorted(args.output.iterdir()):
        with path.open("rb") as stream:
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            }
    receipt = {
        "purpose": "Synthetic graph/state fixture only; no throughput or full-checkpoint quality claim",
        "source": specification["fixture_source"],
        "tokenizer_revision": specification["tokenizer_revision"],
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "tokenizers")
        },
        "files": files,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
