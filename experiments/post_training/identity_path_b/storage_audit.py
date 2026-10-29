# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Path B native-frozen/FP32-trained storage audits; old CD receipt semantics stay intact."""

import hashlib
import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_context_distill.precision_source_audit import (
    CHUNK_BYTES,
    checkpoint_records,
    open_audit_stream,
)
from experiments.post_training.identity_path_b.recipe import special_row_values

FROZEN_HF_NAMES = frozenset(
    f"model.layers.{layer}.{suffix}"
    for layer in range(26)
    for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "mlp.router.weight", "mlp.router.bias")
)


def receipt_tensors(raw: bytes, expected_sha256: str) -> dict:
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Pinned source receipt hash changed")
    tensors = json.loads(raw)["tensors"]
    if len(tensors) != 502 or not FROZEN_HF_NAMES <= tensors.keys():
        raise ValueError("Source receipt must bind502 tensors including exact104 frozen HF names")
    return tensors


def check_source_tensor(name: str, host: np.ndarray, entry: dict) -> dict:
    """Compare frozen native bytes or trainable FP32 master bytes to their own source hash."""
    frozen = name in FROZEN_HF_NAMES
    dtype = "float32" if not frozen or name.endswith(".router.bias") else "bfloat16"
    if str(host.dtype) != dtype or list(host.shape) != entry["shape"]:
        raise ValueError(f"Path B source storage shape/dtype mismatch: {name}")
    if not np.isfinite(host).all():
        raise ValueError(f"Nonfinite Path B source tensor: {name}")
    digest = hashlib.sha256(host.tobytes(order="C")).hexdigest()
    basis = "native_sha256" if frozen else "master_sha256"
    if digest != entry[basis]:
        raise ValueError(f"Path B source bits differ: {name}")
    return {"shape": list(host.shape), "dtype": dtype, "source_hash_basis": basis, "sha256": digest}


def audit_initialized_storage(model, source_receipt: bytes, source_sha256: str, output: str) -> dict | None:
    """Collectively admit exact104 native frozen +398 FP32 trainable source tensors."""
    expected = receipt_tensors(source_receipt, source_sha256)
    tensors = model.to_state_dict()
    if tensors.keys() != expected.keys():
        raise ValueError("Initialized/source full502 tensor keysets differ")
    results = {}
    for name in sorted(tensors):
        replicated = jax.sharding.reshard(tensors[name], PartitionSpec())
        replicated.block_until_ready()

        def check(name, value):
            results[name] = check_source_tensor(name, np.asarray(value.addressable_data(0)), expected[name])

        run_on_export_writer(partial(check, name, replicated))
        del replicated
    report = {
        "schema": "path_b_storage_v1",
        "source_receipt_sha256": source_sha256,
        "frozen_native_tensors": 104,
        "trainable_fp32_tensors": 398,
        "tensors": results,
    }
    run_on_export_writer(lambda: write_export_json(StoragePath(output), report))
    return report if jax.process_index() == 0 else None


def assert_special_row_bits(model, expected: dict) -> dict:
    """Collectively compare original FP32 partial-row bytes, preserving signed zero."""
    actual = special_row_values(model)
    hashes = {}
    for field in sorted(expected):
        if actual[field].dtype != jnp.float32 or expected[field].dtype != jnp.float32:
            raise ValueError("Partial vocabulary rows require FP32 master storage")
        before = jax.sharding.reshard(expected[field], PartitionSpec())
        after = jax.sharding.reshard(actual[field], PartitionSpec())

        def compare(field, before, after):
            left, right = np.asarray(before.addressable_data(0)), np.asarray(after.addressable_data(0))
            if left.tobytes() != right.tobytes():
                raise ValueError(f"Frozen special-row master bits changed: {field}")
            hashes[field] = hashlib.sha256(right.tobytes()).hexdigest()

        run_on_export_writer(partial(compare, field, before, after))
    return hashes


def compare_exported_tensor(name: str, host: np.ndarray, record, source: dict) -> dict:
    """Compare frozen export to pinned source, or trained export to actual master casts."""
    frozen = name in FROZEN_HF_NAMES
    if frozen:
        checked = check_source_tensor(name, host, source)
        if np.dtype(record.dtype) != host.dtype:
            raise ValueError(f"Frozen export dtype changed: {name}")
    else:
        if host.dtype != np.float32 or list(host.shape) != source["shape"]:
            raise ValueError(f"Trainable export master shape/dtype mismatch: {name}")
        if str(np.dtype(record.dtype)) != "bfloat16" or not np.isfinite(host).all():
            raise ValueError(f"Trainable export requires finite FP32 masters and BF16 saved storage: {name}")
        checked = {"master_sha256": hashlib.sha256(host.tobytes()).hexdigest()}
    if tuple(host.shape) != record.shape:
        raise ValueError(f"Export shape changed: {name}")
    flat, digest = host.reshape(-1), hashlib.sha256()
    count = max(1, CHUNK_BYTES // np.dtype(record.dtype).itemsize)
    with open_audit_stream(record.file_path) as stream:
        stream.seek(record.byte_start)
        for start in range(0, flat.size, count):
            expected = flat[start : start + count].astype(record.dtype).tobytes()
            payload = stream.read(len(expected))
            if payload != expected:
                raise ValueError(
                    f"Saved export differs from {'frozen source' if frozen else 'actual master cast'}: {name}"
                )
            digest.update(payload)
    if frozen and digest.hexdigest() != source["native_sha256"]:
        raise ValueError(f"Frozen exported payload differs from pinned source: {name}")
    return {**checked, "frozen_source": frozen, "export_sha256": digest.hexdigest()}


def audit_export_storage(model, export_path: str, source_receipt: bytes, source_sha256: str) -> None:
    """All ranks: verify502 saved tensors with the storage-aware Path B schema."""
    source = receipt_tensors(source_receipt, source_sha256)
    tensors = model.to_state_dict()
    exported = run_on_export_writer(lambda: checkpoint_records(export_path))
    if tensors.keys() != source.keys():
        raise ValueError("Trained model/source full502 keysets differ")

    def check_keys():
        if exported is None or tensors.keys() != exported.keys():
            raise ValueError("Saved export/source full502 keysets differ")

    run_on_export_writer(check_keys)
    results = {}
    for name in sorted(tensors):
        replicated = jax.sharding.reshard(tensors[name], PartitionSpec())
        replicated.block_until_ready()

        def compare(name, value):
            assert exported is not None
            results[name] = compare_exported_tensor(
                name, np.asarray(value.addressable_data(0)), exported[name], source[name]
            )

        run_on_export_writer(partial(compare, name, replicated))
        del replicated
    run_on_export_writer(
        lambda: write_export_json(
            StoragePath(export_path) / "path-b-storage-audit.json",
            {
                "schema": "path_b_storage_v1",
                "source_receipt_sha256": source_sha256,
                "frozen_native_tensors": 104,
                "trained_master_cast_tensors": 398,
                "tensors": results,
            },
        )
    )
