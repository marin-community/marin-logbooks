# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real 32-device CPU explicit-mesh coverage; no simulated sharding or GPU use."""

import os
import subprocess
import sys


def exercise():
    # Imports happen only inside a fresh CPU process after device-count flags.
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    from tempfile import TemporaryDirectory  # noqa: PLC0415

    import equinox as eqx  # noqa: PLC0415
    import haliax as hax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import optax  # noqa: PLC0415
    from jax.sharding import AxisType, Mesh, NamedSharding  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415
    from levanter.checkpoint import save_checkpoint  # noqa: PLC0415
    from levanter.trainer_state import TrainerState  # noqa: PLC0415
    from safetensors.numpy import save_file  # noqa: PLC0415

    from experiments.post_training.identity_context_distill.precision_source_audit import (  # noqa: PLC0415
        checkpoint_records,
    )
    from experiments.post_training.identity_path_b.recipe import (  # noqa: PLC0415
        FROZEN_TOKEN_IDS,
        freeze_special_rows,
        mask_special_rows,
        restore_frozen_state,
        special_row_values,
    )
    from experiments.post_training.identity_path_b.rehydrate import restore_trained_model  # noqa: PLC0415
    from experiments.post_training.identity_path_b.storage_audit import (  # noqa: PLC0415
        assert_special_row_bits,
        compare_exported_tensor,
    )

    class Vocabulary(eqx.Module):
        token_embed: jax.Array
        output_proj: jax.Array

    assert len(jax.devices()) == 32 and all(d.platform == "cpu" for d in jax.devices())
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(2, 4, 4),
        ("replica_dcn", "data", "model"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    ids = np.asarray(FROZEN_TOKEN_IDS)
    host = np.ones((128256, 16), dtype=np.float32)
    host[ids, 0] = np.float32(-0.0)
    with jax.set_mesh(mesh), hax.partitioning.set_mesh(mesh):
        params = Vocabulary(
            jax.device_put(host, NamedSharding(mesh, P("model", ("replica_dcn", "data")))),
            jax.device_put(host.T, NamedSharding(mesh, P(("replica_dcn", "data"), "model"))),
        )
        pristine = jax.tree.map(lambda value: value.copy(), params)
        jax.block_until_ready(pristine)
        snapshots = special_row_values(params)
        for field, value in snapshots.items():
            assert value.sharding.is_fully_replicated, field
            assert value.dtype == jnp.float32
        masked = mask_special_rows(jax.tree.map(jnp.ones_like, params))
        assert np.all(np.asarray(masked.token_embed)[ids] == 0)
        assert np.all(np.asarray(masked.output_proj)[:, ids] == 0)
        assert np.asarray(masked.token_embed)[42, 0] == 1
        optimizer = freeze_special_rows(optax.adamw(learning_rate=0.01, weight_decay=0.1))
        state = TrainerState.init(optimizer, params, key=jax.random.PRNGKey(310))
        grads = jax.tree.map(jnp.ones_like, params)
        step = eqx.filter_jit(lambda state, grads: state.take_step(grads, key=jax.random.PRNGKey(311))[0], donate="all")
        for _ in range(2):
            state = restore_frozen_state(step(state, grads), snapshots)
            assert_special_row_bits(state.model, snapshots)
        assert int(state.step) == 2
        np.testing.assert_array_equal(
            np.asarray(state.model.token_embed)[ids].view(np.uint32), host[ids].view(np.uint32)
        )
        np.testing.assert_array_equal(
            np.asarray(state.model.output_proj)[:, ids].view(np.uint32), host.T[:, ids].view(np.uint32)
        )
        assert np.asarray(state.model.token_embed)[ids[0], 0].view(np.uint32) == 0x80000000
        assert not np.array_equal(np.asarray(state.model.token_embed)[42], host[42])
        assert state.model.token_embed.sharding == params.token_embed.sharding
        assert state.model.output_proj.sharding == params.output_proj.sharding
        with TemporaryDirectory(prefix="pathb-cpu32-") as temporary:
            checkpoint = str(Path(temporary) / "native")
            save_checkpoint(state.saveable_state, step=1, checkpoint_path=checkpoint)
            restored = restore_trained_model(
                pristine,
                True,
                checkpoint,
                completed_updates=2,
                axis_mapping={},
                mesh=mesh,
            )
            assert_special_row_bits(restored, snapshots)
            np.testing.assert_array_equal(
                np.asarray(restored.token_embed)[ids].view(np.uint32), host[ids].view(np.uint32)
            )
            np.testing.assert_array_equal(
                np.asarray(restored.output_proj)[:, ids].view(np.uint32),
                host.T[:, ids].view(np.uint32),
            )
            masters = {
                "model.embed_tokens.weight": np.asarray(restored.token_embed),
                "lm_head.weight": np.asarray(restored.output_proj).T,
            }
            saved = {name: value.astype(jnp.bfloat16) for name, value in masters.items()}
            destination = Path(temporary) / "export"
            destination.mkdir()
            save_file(saved, str(destination / "model.safetensors"))
            (destination / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {name: "model.safetensors" for name in saved},
                        "metadata": {"total_size": sum(value.nbytes for value in saved.values())},
                    }
                )
            )
            for name, record in checkpoint_records(str(destination)).items():
                compare_exported_tensor(name, masters[name], record, {"shape": list(masters[name].shape)})
                assert np.all(saved[name][ids, 0].view(np.uint16) == 0x8000)
    print("CPU32 explicit snapshot/mask/native AdamW/signed-zero restore/export-row assertion PASS")


def test_real_cpu32_explicit_mesh_partial_row_freeze():
    flags = os.environ.get("XLA_FLAGS", "")
    result = subprocess.run(
        [sys.executable, __file__, "--probe"],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_NUM_CPU_DEVICES": "32",
            "XLA_FLAGS": flags + " --xla_force_host_platform_device_count=32",
        },
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU32 explicit" in result.stdout


if __name__ == "__main__":
    exercise()
