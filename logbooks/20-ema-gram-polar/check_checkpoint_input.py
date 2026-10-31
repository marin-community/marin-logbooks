# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise selective checkpoint loading on local arrays, without training."""

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.optim.ema_gram import GramState, gram_direction, init_gram_state, matrix_parallel_eigh, update_gram_state
from optimizer_cost import checkpoint_states, compare_trees


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == "cpu"
    mesh = Mesh(np.array(jax.devices()), ("data",))
    replicated = NamedSharding(mesh, P())
    shapes = [(3, 8, 8)] * 4 + [(3, 32, 8)] * 2 + [(3, 8, 32)]
    rng = np.random.default_rng(923)
    states = []
    bank = "opt_state/inner_state/1/inner_states/muon/inner_state/0/matrices"
    saved = {"step": jnp.array(7, dtype=jnp.int32), "model/unrelated": jnp.ones((8,), dtype=jnp.float32)}
    for i, shape in enumerate(shapes):
        state = init_gram_state(jnp.zeros(shape))
        for _ in range(7):
            gradient = jnp.asarray(rng.standard_normal(shape, dtype=np.float32) * 0.01)
            state = update_gram_state(gradient, state, 0.95, 0.99)
        states.append(state)
        for field, value in zip(GramState._fields, state, strict=True):
            saved[f"{bank}/{i}/{field}"] = value
    save_checkpoint(saved, 7, args.checkpoint_dir)
    loaded, evidence = checkpoint_states(args.checkpoint_dir, shapes, mesh)
    for actual, expected in zip(jax.tree.leaves(loaded), jax.tree.leaves(tuple(states)), strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert actual.sharding == replicated
    comparisons = []
    for ridge in (1e-4, 1e-2):
        for i, state in enumerate(loaded):
            reference = jax.jit(
                partial(gram_direction, alpha=0.25, ridge=ridge, polar_steps=5),
                in_shardings=(replicated,),
                out_shardings=replicated,
            )
            candidate = jax.jit(
                partial(
                    gram_direction,
                    alpha=0.25,
                    ridge=ridge,
                    polar_steps=5,
                    eigh_fn=partial(matrix_parallel_eigh, mesh=mesh, axis_name="data"),
                ),
                in_shardings=(replicated,),
                out_shardings=replicated,
            )
            comparison = compare_trees(jax.block_until_ready(candidate(state)), jax.block_until_ready(reference(state)))
            comparisons.append({"matrix": i, "ridge": ridge, **comparison})
    report = {
        "scope": "Local generated checkpoint on virtual CPU devices; original TPU checkpoint arrays are not read",
        "devices": len(jax.devices()),
        "restored_arrays_bitwise_equal": True,
        "checkpoint": evidence,
        "comparisons": comparisons,
        "passed": all(c["passed"] for c in comparisons),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "comparisons": len(comparisons)}))
    assert report["passed"], "Checkpoint direction comparison failed at the unchanged numerical gate"


if __name__ == "__main__":
    main()
