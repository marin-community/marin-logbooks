# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax
import numpy as np
from jax._src.tpu.linalg.qdwh import qdwh
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

mesh = Mesh(np.array(jax.devices()), ("data",))
rng = np.random.default_rng(1)
x = rng.normal(size=(8, 8, 8)).astype("float32")
x = x @ x.swapaxes(-1, -2) + np.eye(8, dtype="float32") * 0.1
local = jax.vmap(lambda a: qdwh(a)[0])
records = {}
for checked in (True, False):
    fn = jax.jit(
        jax.shard_map(
            local, mesh=mesh, in_specs=P("data", None, None), out_specs=P("data", None, None), check_vma=checked
        )
    )
    try:
        got = np.asarray(fn(x))
        ref = np.asarray(jax.jit(local)(x))
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
        records[str(checked)] = {"passed": True, "max_absolute_error": float(np.max(np.abs(got - ref)))}
    except TypeError as e:
        records[str(checked)] = {"passed": False, "error": str(e)}
assert not records["True"]["passed"] and records["False"]["passed"], records
print(json.dumps(records, indent=2))
