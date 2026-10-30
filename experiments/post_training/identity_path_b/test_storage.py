# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from levanter.adaptor import FrozenParameterAdaptorConfig
from levanter.checkpoint import save_checkpoint
from levanter.checkpoint_manifest import read_manifest, write_manifest
from levanter.compat.fsspec_safetensor import TensorRecord
from levanter.trainer_state import FrozenParameterStorage, TrainerState
from safetensors.numpy import save_file

from experiments.post_training.identity_context_distill.precision_source_audit import checkpoint_records
from experiments.post_training.identity_path_b.recipe import (
    FROZEN_FIELDS,
    FROZEN_TOKEN_IDS,
    freeze_special_rows,
    restore_frozen_state,
    special_row_values,
)
from experiments.post_training.identity_path_b.rehydrate import restore_trained_model
from experiments.post_training.identity_path_b.storage_audit import (
    FROZEN_HF_NAMES,
    assert_special_row_bits,
    audit_export_storage,
    audit_initialized_storage,
    check_source_tensor,
    compare_exported_tensor,
)
from experiments.post_training.identity_path_b.testing_support import Layer, StorageModel


def test_finite_fp32_master_overflow_to_matching_bf16_infinity_is_rejected(tmp_path):
    host = np.asarray([np.finfo(np.float32).max, 1.0], dtype=np.float32)
    with np.errstate(over="ignore"):
        saved = host.astype(jnp.bfloat16)
    assert np.isfinite(host).all() and not np.isfinite(saved).all()
    path = tmp_path / "overflow.bin"
    path.write_bytes(saved.tobytes())
    record = TensorRecord("trainable.overflow.weight", np.dtype(jnp.bfloat16), host.shape, str(path), 0, saved.nbytes)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="Nonfinite cast/output"):
        compare_exported_tensor(record.key, host, record, {"shape": list(host.shape)})


@pytest.mark.timeout(180)
def test_real_update_native_checkpoint_rehydrates_exact_frozen_source_then_audits_export(tmp_path):
    def scalar(dtype):
        return jax.device_put(np.asarray([-0.0, 1.0], dtype=dtype))

    ids = jnp.asarray(FROZEN_TOKEN_IDS)
    source = StorageModel(
        jnp.ones((128256, 1)).at[ids, 0].set(-0.0),
        jnp.ones((1, 128256)).at[0, ids].set(-0.0),
        [Layer(scalar(jnp.bfloat16), scalar(jnp.bfloat16), scalar(jnp.bfloat16), scalar(np.float32)) for _ in range(26)],
        {f"trainable.{index}.weight": scalar(np.float32) for index in range(396)},
    )
    tensors = source.to_state_dict()
    entries = {}
    for name, array in tensors.items():
        host = np.asarray(array)
        native = host if name in FROZEN_HF_NAMES else host.astype(jnp.bfloat16)
        entries[name] = {
            "shape": list(host.shape),
            "native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
            "master_sha256": hashlib.sha256(host.astype(np.float32).tobytes()).hexdigest(),
        }
    raw = json.dumps({"tensors": entries}).encode()
    source_sha = hashlib.sha256(raw).hexdigest()
    adapter = FrozenParameterAdaptorConfig(frozen_field_names=FROZEN_FIELDS)
    mask = adapter.trainable_filter(source)
    optimizer = freeze_special_rows(optax.adamw(learning_rate=0.01, weight_decay=0.1))
    state = TrainerState.init(
        optimizer,
        source,
        key=jax.random.PRNGKey(310),
        is_trainable=mask,
        mp=jmp.get_policy("p=f32,c=bfloat16,o=f32"),
        frozen_parameter_storage=FrozenParameterStorage.NATIVE,
    )
    pristine = jax.tree.map(lambda array: array.copy(), state.model)
    jax.block_until_ready(pristine)
    partial = special_row_values(state.model)
    grads = jax.tree.map(jnp.ones_like, state.trainable_model)
    take_step = eqx.filter_jit(lambda state, grads: state.take_step(grads, key=jax.random.PRNGKey(311))[0], donate="all")
    state = restore_frozen_state(take_step(state, grads), partial)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    checkpoint = str(tmp_path / "native")
    with hax.partitioning.set_mesh(mesh):
        admitted = audit_initialized_storage(pristine, raw, source_sha, str(tmp_path / "source-admission.json"))
        assert admitted is not None
        assert admitted["frozen_native_tensors"] == 104 and admitted["trainable_fp32_tensors"] == 398
        save_checkpoint(state.saveable_state, step=0, checkpoint_path=checkpoint)
        restored = restore_trained_model(pristine, mask, checkpoint, completed_updates=1, axis_mapping={}, mesh=mesh)
        assert_special_row_bits(restored, partial)
        np.testing.assert_array_equal(
            np.asarray(restored.token_embed)[np.asarray(ids)].view(np.uint32),
            np.asarray(pristine.token_embed)[np.asarray(ids)].view(np.uint32),
        )
        assert np.asarray(pristine.token_embed)[int(ids[0]), 0].view(np.uint32) == 0x80000000
        assert not np.array_equal(restored.token_embed[42], pristine.token_embed[42])
        for name in FROZEN_HF_NAMES:
            assert (
                np.asarray(restored.to_state_dict()[name]).tobytes()
                == np.asarray(pristine.to_state_dict()[name]).tobytes()
            )
        destination = tmp_path / "export"
        destination.mkdir()
        saved = {
            name: np.asarray(array).astype(np.float32 if name.endswith(".router.bias") else jnp.bfloat16)
            for name, array in restored.to_state_dict().items()
        }
        save_file(saved, str(destination / "model.safetensors"))
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {name: "model.safetensors" for name in saved},
                    "metadata": {"total_size": sum(array.nbytes for array in saved.values())},
                }
            )
        )
        audit_export_storage(restored, str(destination), raw, source_sha)
        proof = json.loads((destination / "path-b-storage-audit.json").read_text())
        assert len(proof["tensors"]) == 502
        assert sum(row["frozen_source"] for row in proof["tensors"].values()) == 104
        q = "model.layers.0.self_attn.q_proj.weight"
        with pytest.raises(ValueError, match="source bits differ"):
            check_source_tensor(q, np.asarray(pristine.layers[0].w_q.at[1].set(2)), entries[q])
        manifest = read_manifest(checkpoint)
        assert manifest is not None
        model_entry = next(entry for entry in manifest.arrays if entry.path == "model/token_embed")
        incomplete = manifest.model_copy(
            update={"arrays": tuple(entry for entry in manifest.arrays if entry.path != model_entry.path)}
        )
        write_manifest(checkpoint, incomplete)
        with pytest.raises(ValueError, match="missing, extra"):
            restore_trained_model(pristine, mask, checkpoint, completed_updates=1, axis_mapping={}, mesh=mesh)
        overlap = model_entry.model_copy(update={"path": "model/layers/0/w_q", "shape": (2,), "dtype": "bfloat16"})
        write_manifest(checkpoint, manifest.model_copy(update={"arrays": (*manifest.arrays, overlap)}))
        with pytest.raises(ValueError, match="frozen-overlap"):
            restore_trained_model(pristine, mask, checkpoint, completed_updates=1, axis_mapping={}, mesh=mesh)
        write_manifest(checkpoint, manifest)
        assert len(checkpoint_records(str(destination))) == 502
