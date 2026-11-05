# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a native Grug export and compare two weights with the initializer."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for

WEIGHTS = ("model.layers.0.mlp.router.weight", "model.layers.0.self_attn.o_proj.weight")


def read_bytes(uri: str) -> bytes:
    filesystem, key = filesystem_for(uri)
    return filesystem.cat_file(key)


def tensor(model: str, index: dict, name: str) -> tuple[np.ndarray, list[int]]:
    """Read one BF16 tensor by byte range without fetching its entire shard."""
    filesystem, key = filesystem_for(model + "/" + index["weight_map"][name])
    length = struct.unpack("<Q", filesystem.cat_file(key, start=0, end=8))[0]
    entry = json.loads(filesystem.cat_file(key, start=8, end=8 + length))[name]
    if entry["dtype"] != "BF16":
        raise ValueError("The pinned initializer/export dtype changed")
    start, end = entry["data_offsets"]
    payload = filesystem.cat_file(key, start=8 + length + start, end=8 + length + end)
    value = (np.frombuffer(payload, dtype="<u2").astype("<u4") << 16).view("<f4")
    if value.size != np.prod(entry["shape"]) or not np.isfinite(value).all():
        raise ValueError("Incomplete or nonfinite exported tensor")
    return value, entry["shape"]


def verify(
    root: str,
    initial: str,
    revision: str,
    terminal_uri: str | None = None,
    expected_job_id: str | None = None,
) -> dict:
    """Verify native completion, the shard inventory and sampled weight changes."""
    temporary = root.replace("s3://marin-rno2a/marin/", "s3://marin-rno2a/tmp/ttl=14d/skyrl/marin/", 1)
    request = json.loads(read_bytes(temporary + "/checkpoints/global_step_50/hf_export_request.json"))
    terminal_uri = terminal_uri or temporary + "/attempts/export-50/terminal.json"
    terminal = json.loads(read_bytes(terminal_uri))["result"]
    if request["status"] != "complete" or request["last_exit_code"] != 0 or terminal["state"] != "succeeded":
        raise ValueError("Native export did not finish successfully")
    if expected_job_id is not None and terminal["iris_job_id"] != expected_job_id:
        raise ValueError("Native terminal record belongs to a different export job")
    model = root + "/exports/global_step_50/policy"
    manifest = read_bytes(model + "/.marinskyrl-model-manifest.json")
    index = json.loads(read_bytes(model + "/model.safetensors.index.json"))
    initial_index = json.loads(read_bytes(initial + "/model.safetensors.index.json"))
    filesystem, prefix = filesystem_for(model)
    files = filesystem.find(prefix, detail=True)
    shards = {key: info["size"] for key, info in files.items() if key.endswith(".safetensors")}
    expected = {prefix.rstrip("/") + "/" + name for name in index["weight_map"].values()}
    if set(shards) != expected or len(shards) != 39 or not all(size > 0 for size in shards.values()):
        raise ValueError("The complete 39-shard portable inventory is required")
    checks = []
    for name in WEIGHTS:
        before, shape = tensor(initial, initial_index, name)
        after, final_shape = tensor(model, index, name)
        if shape != final_shape:
            raise ValueError("Initial and exported tensor shapes differ")
        checks.append(
            {
                "name": name,
                "shape": shape,
                "initial_float32_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
                "final_float32_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
                "changed_elements": int(np.count_nonzero(before != after)),
                "elements": before.size,
                "max_abs_change": float(np.max(np.abs(before - after))),
            }
        )
    if not any(row["changed_elements"] for row in checks):
        raise ValueError("Neither sampled tensor changed from the initializer")
    return {
        "passed": True,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_source_revision": revision,
        "run_root": root,
        "initial_model": initial,
        "export_request_status": request["status"],
        "export_request_last_exit_code": request["last_exit_code"],
        "export_terminal_state": terminal["state"],
        "export_terminal_uri": terminal_uri,
        "export_iris_job_id": terminal["iris_job_id"],
        "model_export": model,
        "model_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "weight_shards": len(shards),
        "weight_shard_bytes": sum(shards.values()),
        "sampled_weight_checks": checks,
        "limits": (
            "Two exported tensors compared with the exact initial checkpoint. This does not check every parameter "
            "or live inference replica; Qwen separately qualified live restoration. "
            "The native converter verifies the portable inventory."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("initial_model")
    parser.add_argument("source_revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("--terminal-uri", help="Successful recovery attempt's own native terminal record")
    parser.add_argument("--expected-job-id", help="Require the terminal record to identify this exact export job")
    args = parser.parse_args()
    proof = verify(
        args.run_root.rstrip("/"),
        args.initial_model.rstrip("/"),
        args.source_revision,
        args.terminal_uri,
        args.expected_job_id,
    )
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({key: proof[key] for key in ("passed", "weight_shards", "weight_shard_bytes")}))
