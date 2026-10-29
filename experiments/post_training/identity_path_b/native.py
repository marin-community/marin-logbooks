# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load explicitly segmented packed examples through the native Grug boundary."""

import hashlib
import io

import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from levanter.data.dataset import ListAsyncDataset, MappedAsyncDataset
from levanter.data.text.datasets import NamedLmDataset
from levanter.data.text.examples import GrugLmExample


def packed_example(row: dict) -> GrugLmExample:
    """Shift weights by the predicted token once; preserve document/pad segments."""
    tokens = jnp.asarray(row["input_ids"], dtype=jnp.int32)
    target_weights = jnp.asarray(row["target_weights"], dtype=jnp.float32)
    segments = jnp.asarray(row["segment_ids"], dtype=jnp.int32)
    return GrugLmExample.causal(
        tokens=tokens,
        loss_weight=jnp.roll(target_weights, -1),
        segment_ids=segments,
        block_cross_document_attention=True,
    )


def packed_dataset(raw: bytes, expected_sha256: str, pos):
    """Bind exact regional parquet bytes, with no runtime sampling or reshuffling."""
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Packed parquet differs from the reviewed regional bytes")
    table = pq.read_table(io.BytesIO(raw))
    records = table.to_pylist()
    for row in records:
        ids, weights, segments = (np.asarray(row[k]) for k in ["input_ids", "target_weights", "segment_ids"])
        if ids.shape != (pos.size,) or weights.shape != ids.shape or segments.shape != ids.shape:
            raise ValueError("Packed sequence has the wrong dimensions")
        if np.any(weights[segments < 0] != 0):
            raise ValueError("Packed padding carries loss")
        if np.any((weights != 0) & (weights != np.float32(0.1)) & (weights != 1)):
            raise ValueError("Unexpected nonbinary/special-token weight")
    native = MappedAsyncDataset(ListAsyncDataset(records), packed_example)
    return NamedLmDataset(native, pos)
