# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.adaptor import FrozenParameterAdaptorConfig
from levanter.checkpoint import save_checkpoint
from levanter.optim.config import AdamConfig
from levanter.trainer_state import FrozenParameterStorage, TrainerState
from safetensors.numpy import save_file

from experiments.post_training.identity_path_b.ablation import (
    AblationArm,
    freezes_special_rows,
    frozen_fields,
    frozen_names,
    optimizer_for,
    updates_for,
)
from experiments.post_training.identity_path_b.recipe import FROZEN_TOKEN_IDS, restore_frozen_state, special_row_values
from experiments.post_training.identity_path_b.rehydrate import restore_trained_model
from experiments.post_training.identity_path_b.storage_audit import audit_export_storage, audit_initialized_storage
from experiments.post_training.identity_path_b.testing_support import Layer, StorageModel


@pytest.mark.parametrize(
    "arm", [AblationArm.CONTROL, AblationArm.TRAIN_SPECIAL_ROWS, AblationArm.T3_BRIDGE, AblationArm.GLM_ON_TENTH_72]
)
@pytest.mark.timeout(180)
def test_actual_update_restore_and_502_tensor_audit_obey_each_freeze_variant(tmp_path, arm):
    def scalar(dtype):
        return jnp.asarray([-0.0, 1.0], dtype=dtype)

    ids = jnp.asarray(FROZEN_TOKEN_IDS)
    source = StorageModel(
        jnp.ones((128256, 1)).at[ids, 0].set(-0.0),
        jnp.ones((1, 128256)).at[0, ids].set(-0.0),
        [
            Layer(scalar(jnp.bfloat16), scalar(jnp.bfloat16), scalar(jnp.bfloat16), scalar(jnp.float32))
            for _ in range(26)
        ],
        {f"trainable.{index}.weight": scalar(jnp.float32) for index in range(396)},
    )
    entries = {}
    for name, value in source.to_state_dict().items():
        host = np.asarray(value)
        native = host.astype(np.float32 if name.endswith(".router.bias") else jnp.bfloat16)
        entries[name] = {
            "shape": list(host.shape),
            "native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
            "master_sha256": hashlib.sha256(host.astype(np.float32).tobytes()).hexdigest(),
        }
    raw = json.dumps({"tensors": entries}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    optimizer = optimizer_for(
        SimpleNamespace(optimizer=AdamConfig(learning_rate=0.01, warmup=0, lr_schedule="constant")), arm
    )
    mask = FrozenParameterAdaptorConfig(frozen_field_names=frozen_fields(arm)).trainable_filter(source)
    state = TrainerState.init(
        optimizer,
        source,
        key=jax.random.PRNGKey(310),
        is_trainable=mask,
        mp=jmp.get_policy("p=f32,c=bfloat16,o=f32"),
        frozen_parameter_storage=FrozenParameterStorage.NATIVE,
    )
    pristine = jax.tree.map(lambda value: value.copy(), state.model)
    special = special_row_values(state.model) if freezes_special_rows(arm) else None
    grads = jax.tree.map(jnp.ones_like, state.trainable_model)
    step = eqx.filter_jit(lambda state, grads: state.take_step(grads, key=jax.random.PRNGKey(311))[0], donate="all")
    state = step(state, grads)
    if special is not None:
        state = restore_frozen_state(state, special)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    with hax.partitioning.set_mesh(mesh):
        audit_initialized_storage(
            pristine, raw, digest, str(tmp_path / "initialized.json"), frozen_names=frozen_names(arm)
        )
        native = str(tmp_path / "native")
        save_checkpoint(state.saveable_state, step=0, checkpoint_path=native)
        restored = restore_trained_model(pristine, mask, native, completed_updates=1, axis_mapping={}, mesh=mesh)
        before = np.asarray(pristine.token_embed)[np.asarray(ids)].view(np.uint32)
        after = np.asarray(restored.token_embed)[np.asarray(ids)].view(np.uint32)
        if freezes_special_rows(arm):
            np.testing.assert_array_equal(before, after)
        else:
            assert not np.array_equal(before, after)
        tensors = restored.to_state_dict()
        for name in frozen_names(arm):
            assert np.asarray(tensors[name]).tobytes() == np.asarray(pristine.to_state_dict()[name]).tobytes()
        query = "model.layers.0.self_attn.q_proj.weight"
        assert (np.asarray(tensors[query]).tobytes() == np.asarray(pristine.to_state_dict()[query]).tobytes()) == (
            arm != AblationArm.T3_BRIDGE
        )
        destination = tmp_path / "hf"
        destination.mkdir()
        saved = {
            name: np.asarray(value).astype(np.float32 if name.endswith(".router.bias") else jnp.bfloat16)
            for name, value in tensors.items()
        }
        save_file(saved, str(destination / "model.safetensors"))
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {name: "model.safetensors" for name in saved},
                    "metadata": {"total_size": sum(value.nbytes for value in saved.values())},
                }
            )
        )
        audit_export_storage(restored, str(destination), raw, digest, frozen_names=frozen_names(arm))
        report = json.loads((destination / "path-b-storage-audit.json").read_text())
        assert report["frozen_native_tensors"] == len(frozen_names(arm))
        assert report["trained_master_cast_tensors"] == 502 - len(frozen_names(arm))
        assert {name for name, row in report["tensors"].items() if row["frozen_source"]} == frozen_names(arm)


def test_A9_extends_constant_trajectory_without_restarting_warmup():
    config = AdamConfig(learning_rate=2e-6, warmup=2, lr_schedule="constant")
    old = config.lr_scheduler(updates_for(AblationArm.GLM_ON_TENTH))
    extended = config.lr_scheduler(updates_for(AblationArm.GLM_ON_TENTH_72))
    np.testing.assert_array_equal([float(old(i)) for i in range(72)], [float(extended(i)) for i in range(72)])
    assert float(extended(0)) == 0 and float(extended(1)) > 0
    assert float(extended(36)) == float(extended(71)) == float(extended(2))
