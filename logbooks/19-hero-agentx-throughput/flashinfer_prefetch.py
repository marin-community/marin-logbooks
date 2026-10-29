# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch the pinned FMHA metadata once before concurrent worker startup."""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from flashinfer.artifacts import ArtifactPath, CheckSumHash
from flashinfer.jit.attention.modules import get_artifact, get_meta_hash

assert importlib.metadata.version("flashinfer-python") == "0.6.18.post1"
directory = ArtifactPath.TRTLLM_GEN_FMHA
checksums = get_artifact(f"{directory}/checksums.txt", CheckSumHash.TRTLLM_GEN_FMHA)
assert checksums
expected = get_meta_hash(checksums)
header = get_artifact(f"{directory}/include/flashInferMetaInfo.h", expected)
assert header and hashlib.sha256(header).hexdigest() == expected
record = {
    "flashinfer_version": importlib.metadata.version("flashinfer-python"),
    "checksums_sha256": hashlib.sha256(checksums).hexdigest(),
    "header_sha256": expected,
    "header_bytes": len(header),
    "scope": "Native artifact fetch and hash verification before workers; no GPU kernels or arithmetic changes.",
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
