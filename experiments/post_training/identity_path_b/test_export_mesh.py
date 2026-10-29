# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real CPU32 -> CPU8 native restore with all502 storage audit roles; no weights/GPU."""

import os
import subprocess
import sys

import pytest


def exercise():
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    from tempfile import TemporaryDirectory  # noqa: PLC0415

    import equinox as eqx  # noqa: PLC0415
    import haliax as hax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import jmp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import optax  # noqa: PLC0415
    from jax.sharding import AxisType, Mesh, NamedSharding  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415
    from levanter.adaptor import FrozenParameterAdaptorConfig  # noqa: PLC0415
    from levanter.checkpoint import CheckpointerConfig  # noqa: PLC0415
    from levanter.trainer_state import FrozenParameterStorage, TrainerState  # noqa: PLC0415
    from safetensors.numpy import save_file  # noqa: PLC0415

    from experiments.post_training.identity_path_b.recipe import (  # noqa: PLC0415
        FROZEN_FIELDS,
        FROZEN_TOKEN_IDS,
        special_row_values,
    )
    from experiments.post_training.identity_path_b.rehydrate import (  # noqa: PLC0415
        restore_trained_model,
        source_restore_template,
    )
    from experiments.post_training.identity_path_b.retention import save_temporary_checkpoint  # noqa: PLC0415
    from experiments.post_training.identity_path_b.storage_audit import (  # noqa: PLC0415
        FROZEN_HF_NAMES,
        assert_special_row_bits,
        audit_export_storage,
        audit_initialized_storage,
    )
    from experiments.post_training.identity_path_b.test_storage import Layer, StorageModel  # noqa: PLC0415
    from experiments.post_training.identity_path_b.worker import export_mesh  # noqa: PLC0415

    devices = jax.devices()
    assert len(devices) == 32 and all(d.platform == "cpu" for d in devices)
    axes = ("data", "expert", "replica", "model", "context", "replica_dcn")
    train_mesh = Mesh(np.asarray(devices).reshape(2, 4, 1, 1, 4, 1), axes, axis_types=(AxisType.Explicit,) * 6)
    layout = export_mesh()
    ici, dcn = layout.axis_shapes(8, 1)
    assert ici["context"] == 1 and ici["expert"] == 4 and ici["data"] == 2 and dcn == {"replica_dcn": 1}
    small_mesh = Mesh(np.asarray(devices[:8]).reshape(2, 4, 1, 1, 1, 1), axes, axis_types=(AxisType.Explicit,) * 6)
    ids = np.asarray(FROZEN_TOKEN_IDS)
    embedding = np.ones((128256, 8), dtype=np.float32)
    embedding[ids, 0] = np.float32(-0.0)
    head = embedding.T.copy()
    small = np.asarray([-0.0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    host_model = StorageModel(
        embedding,
        head,
        [
            Layer(small.astype(jnp.bfloat16), small.astype(jnp.bfloat16), small.astype(jnp.bfloat16), small)
            for _ in range(26)
        ],
        {f"trainable.{i}.weight": small for i in range(396)},
    )

    def place(model, mesh):
        def put(array):
            spec = (
                P(None, ("data", "context"))
                if array.shape == head.shape
                else P(("data", "context"), *([None] * (array.ndim - 1)))
            )
            return jax.device_put(array, NamedSharding(mesh, spec))

        return jax.tree.map(put, model)

    adapter = FrozenParameterAdaptorConfig(frozen_field_names=FROZEN_FIELDS)
    trainable = adapter.trainable_filter(host_model)
    # Small deterministic trained masters, preserving special rows and all fully frozen fields.
    changed = jax.tree.map(lambda a: np.asarray(a) + np.float32(0.25), eqx.filter(host_model, trainable))
    trained_host = eqx.combine(changed, host_model)
    trained_host = eqx.tree_at(
        lambda m: (m.token_embed, m.output_proj),
        trained_host,
        (np.array(trained_host.token_embed), np.array(trained_host.output_proj)),
    )
    trained_host.token_embed[ids] = embedding[ids]
    trained_host.output_proj[:, ids] = head[:, ids]
    entries = {}
    for name, array in host_model.to_state_dict().items():
        host = np.asarray(array)
        native = host if name in FROZEN_HF_NAMES else host.astype(jnp.bfloat16)
        entries[name] = {
            "shape": list(host.shape),
            "native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
            "master_sha256": hashlib.sha256(host.astype(np.float32).tobytes()).hexdigest(),
        }
    raw = json.dumps({"tensors": entries}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with TemporaryDirectory() as temp:
        checkpoint = str(Path(temp) / "native")
        with hax.partitioning.set_mesh(train_mesh):
            model = place(trained_host, train_mesh)
            state = TrainerState.init(
                optax.sgd(0.01),
                model,
                key=jax.random.PRNGKey(310),
                is_trainable=adapter.trainable_filter(model),
                mp=jmp.get_policy("p=f32,c=bfloat16,o=f32"),
                frozen_parameter_storage=FrozenParameterStorage.NATIVE,
            )
            writer = CheckpointerConfig(
                base_path=checkpoint,
                append_run_id_to_base_path=False,
                save_interval=None,
                keep=None,
                keep_last_temporary_checkpoints=2,
                delete_old_temp_checkpoints=False,
            ).create("fixture")
            save_temporary_checkpoint(writer, state.saveable_state, step=6)
            writer.wait_until_finished()
        native = checkpoint + "/step-6"
        assert json.loads(Path(native, "metadata.json").read_text())["is_temporary"] is True
        with hax.partitioning.set_mesh(small_mesh):
            pristine = place(host_model, small_mesh)
            audit_initialized_storage(pristine, raw, digest, str(Path(temp) / "source.json"))
            rows = special_row_values(pristine)
            template = source_restore_template(pristine, adapter.trainable_filter(pristine))
            assert all(
                isinstance(a, jax.ShapeDtypeStruct)
                for a in jax.tree.leaves(eqx.filter(template, adapter.trainable_filter(pristine)))
            )
            restored = restore_trained_model(
                template,
                adapter.trainable_filter(pristine),
                native,
                completed_updates=7,
                axis_mapping=layout.resolved_param_mapping,
                mesh=small_mesh,
            )
            assert_special_row_bits(restored, rows)
            for name, array in restored.to_state_dict().items():
                assert array.sharding.device_set <= set(devices[:8])
                assert np.asarray(array).tobytes() == np.asarray(trained_host.to_state_dict()[name]).tobytes(), name
            np.testing.assert_array_equal(
                np.asarray(restored.token_embed)[ids].view(np.uint32), embedding[ids].view(np.uint32)
            )
            np.testing.assert_array_equal(
                np.asarray(restored.output_proj)[:, ids].view(np.uint32), head[:, ids].view(np.uint32)
            )
            exported = Path(temp) / "hf"
            exported.mkdir()
            arrays = {
                name: np.asarray(a).astype(np.float32 if name.endswith(".router.bias") else jnp.bfloat16)
                for name, a in restored.to_state_dict().items()
            }
            save_file(arrays, str(exported / "model.safetensors"))
            (exported / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {name: "model.safetensors" for name in arrays},
                        "metadata": {"total_size": sum(a.nbytes for a in arrays.values())},
                    }
                )
            )
            audit_export_storage(restored, str(exported), raw, digest)
            audit = json.loads((exported / "path-b-storage-audit.json").read_text())
            assert (
                len(audit["tensors"]) == 502
                and audit["frozen_native_tensors"] == 104
                and audit["trained_master_cast_tensors"] == 398
            )
    print("CPU32-to-CPU8 TEMP native restore /502audit/104native/398master/signedzero PASS")


@pytest.mark.timeout(240)
def test_native_checkpoint_restores_identical_bits_on_eight_device_export_mesh():
    result = subprocess.run(
        [sys.executable, __file__, "--probe"],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_NUM_CPU_DEVICES": "32",
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32",
        },
        capture_output=True,
        text=True,
        timeout=210,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU32-to-CPU8" in result.stdout


if __name__ == "__main__":
    exercise()
