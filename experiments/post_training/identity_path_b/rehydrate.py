# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overlay only the exact native checkpoint trainable set onto an admitted pinned source."""

import json

import equinox as eqx
import jax
from haliax.util import is_named_array
from levanter.checkpoint import load_checkpoint
from levanter.checkpoint_manifest import read_manifest
from levanter.utils.jax_utils import leaf_key_paths
from rigging.filesystem.storage_path import StoragePath


def restore_trained_model(
    source_model, trainable_filter, checkpoint_path: str, *, completed_updates: int, axis_mapping, mesh
):
    """Require exact model paths/shapes/dtypes; rehydrate omitted frozen fields from source.

    The caller must first admit the full source with Path B's storage-aware audit.
    Native TrainerState checkpoints also contain optimizer/progress arrays; those
    are outside model/ and are intentionally not restored for independent export.
    """
    metadata = json.loads((StoragePath(checkpoint_path) / "metadata.json").read_bytes())
    if metadata["step"] != completed_updates - 1:
        raise ValueError("Native checkpoint label differs from the completed-update pin")
    manifest = read_manifest(checkpoint_path)
    if manifest is None:
        raise ValueError("Path B independent export requires the committed native array manifest")
    trainable = eqx.filter(source_model, trainable_filter)
    tree = {"model": trainable}
    names = jax.tree.leaves(leaf_key_paths(tree, is_leaf=is_named_array))
    arrays = jax.tree.leaves(tree, is_leaf=is_named_array)
    expected = {}
    for name, array in zip(names, arrays, strict=True):
        raw = array.array if is_named_array(array) else array
        expected[name.replace(".", "/")] = (tuple(raw.shape), str(raw.dtype))
    entries = [entry for entry in manifest.arrays if entry.path.startswith("model/")]
    actual = {entry.path: (entry.shape, entry.dtype) for entry in entries}
    if len(actual) != len(entries) or actual != expected:
        raise ValueError("Checkpoint model set/shape/dtype differs: missing, extra, duplicate or frozen-overlap key")
    restored = load_checkpoint(tree, checkpoint_path, axis_mapping=axis_mapping, mesh=mesh, allow_partial=False)
    return eqx.combine(restored["model"], source_model)


def source_restore_template(source_model, trainable_filter):
    """Release admitted trainable source values before allocating trained masters.

    Preserve exact shape/dtype/sharding exemplars and every frozen source value.
    This changes export memory lifetime only; the complete source audit and partial
    row snapshot must precede this call. Native load still requires every key.
    """
    return jax.tree.map(
        lambda value, trainable: (
            jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding) if trainable else value
        ),
        source_model,
        trainable_filter,
    )
