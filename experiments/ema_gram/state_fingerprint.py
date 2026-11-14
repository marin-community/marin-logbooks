# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact local-shard identities without retaining accelerator copies."""

import hashlib

import numpy as np
from levanter.tensorstore_serialization import _flatten_serializable_leaves


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def describe_arrays(tree):
    """Keep local shard indices explicit; hashes are not cross-sharding comparisons."""
    paths, arrays = _flatten_serializable_leaves(tree)
    result = []
    for path, array in zip(paths, arrays, strict=True):
        for shard in array.addressable_shards:
            host = np.asarray(shard.data)
            assert np.isfinite(host).all(), path
            result.append(dict(path=path, index=str(shard.index), shape=list(host.shape), sha256=digest(host)))
    return result
