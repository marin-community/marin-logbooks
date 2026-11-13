# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned Snowball Hugging Face weights into a native checkpoint the checkpoint eval restores.

A port of the cold-start reproduction's importer (marin#10799) to ``moe_hero_ep.model``. It keeps that
importer's tensor mapping and its exact-parity check through the independent export mapping. It writes
``{"step", "params"}`` with the shared expert as the model holds it, which is what
``launch_checkpoint_eval.load_checkpoint_params`` reads; the reproduction's trainer instead initialized
from a legacy layout that this file does not write.

Run it as one process that owns every GPU of a node; the source and the model are each about 134 GB.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import replace

import click
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig
from levanter.utils.jax_utils import leaf_key_paths
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.launch_checkpoint_eval import load_checkpoint_params
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer, grugmoe_inference_state_dict
from experiments.grug.moe_hero_ep.snowball_recipe import SNOWBALL_67B_MODEL_CONFIG, SNOWBALL_MIXED_PRECISION

logger = logging.getLogger(__name__)

CHECKPOINT_STEP = 0
SHARED_EXPERT_LIST_KEY = ".shared_experts.0."
SHARED_EXPERT_KEY = ".shared_expert."
SAVE_TIMEOUT = 1800
# Native leaf path -> (Hugging Face tensor name, whether the linear weight is stored transposed).
ROOT_WEIGHTS = {
    "token_embed": ("model.embed_tokens.weight", False),
    "embed_norm.weight": ("model.embed_norm.weight", False),
    "embed_gated_norm.w_down": ("model.embed_gated_norm.down_proj.weight", True),
    "embed_gated_norm.w_up": ("model.embed_gated_norm.up_proj.weight", True),
    "output_proj": ("lm_head.weight", True),
    "final_norm.weight": ("model.norm.weight", False),
    "final_gated_norm.w_down": ("model.final_gated_norm.down_proj.weight", True),
    "final_gated_norm.w_up": ("model.final_gated_norm.up_proj.weight", True),
}
BLOCK_PREFIX = "stacked_blocks.stacked."
BLOCK_WEIGHTS = {
    "rms_attn.weight": ("input_layernorm.weight", False),
    "attn_gated_norm.w_down": ("attn_gated_norm.down_proj.weight", True),
    "attn_gated_norm.w_up": ("attn_gated_norm.up_proj.weight", True),
    "attn.w_q": ("self_attn.q_proj.weight", True),
    "attn.w_k": ("self_attn.k_proj.weight", True),
    "attn.w_v": ("self_attn.v_proj.weight", True),
    "attn.w_o": ("self_attn.o_proj.weight", True),
    "attn.attn_gate": ("self_attn.attn_gate.weight", True),
    "rms_mlp.weight": ("post_attention_layernorm.weight", False),
    "mlp_gated_norm.w_down": ("mlp_gated_norm.down_proj.weight", True),
    "mlp_gated_norm.w_up": ("mlp_gated_norm.up_proj.weight", True),
    "mlp.router": ("mlp.router.weight", True),
    "mlp.router_bias": ("mlp.router.bias", False),
    "mlp.expert_mlp.w_gate": ("mlp.experts.gate_proj.weight", True),
    "mlp.expert_mlp.w_up": ("mlp.experts.up_proj.weight", True),
    "mlp.expert_mlp.w_down": ("mlp.experts.down_proj.weight", True),
    "shared.0.w_gate": ("shared_expert.gate_proj.weight", True),
    "shared.0.w_up": ("shared_expert.up_proj.weight", True),
    "shared.0.w_down": ("shared_expert.down_proj.weight", True),
}


def import_weights(config: GrugModelConfig, source: Mapping[str, jax.Array]) -> Transformer:
    """Map every Hugging Face tensor into the stacked model; fail on a shape mismatch or an unmapped tensor."""
    if config.num_shared_experts != 1 or config.latent_dim is not None or config.sconv:
        raise ValueError("This importer requires the original Snowball architecture")
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.key(0))
    consumed = set()

    def load(path, target):
        stacked = path.startswith(BLOCK_PREFIX)
        if stacked:
            suffix, transpose = BLOCK_WEIGHTS[path.removeprefix(BLOCK_PREFIX)]
            names = [f"model.layers.{i}.{suffix}" for i in range(config.num_layers)]
        else:
            name, transpose = ROOT_WEIGHTS[path]
            names = [name]
        values = [jnp.swapaxes(source[name], -1, -2) if transpose else source[name] for name in names]
        value = jnp.stack(values) if stacked else values[0]
        if value.shape != target.shape:
            raise ValueError(f"Shape mismatch for {path}: {value.shape} != {target.shape}")
        consumed.update(names)
        return jax.sharding.reshard(value, target.sharding)

    model = jax.tree.map(load, leaf_key_paths(template), template)
    if consumed != set(source):
        raise ValueError(f"Unmapped HF weights: {sorted(set(source) - consumed)}")
    return model


def verify_weights(model: Transformer, source: Mapping[str, jax.Array]) -> list[dict]:
    """Compare every value through the independent export mapping, one layer at a time to bound memory."""
    errors = []
    records = []
    checked = set()
    for layer_index in range(model.config.num_layers):
        stacked = jax.tree.map(lambda a, i=layer_index: a[i : i + 1], model.stacked_blocks.stacked)
        # The export walks ``config.num_layers``, so the one-layer view needs a one-layer config as well.
        layer_model = replace(
            model,
            config=replace(model.config, num_layers=1),
            stacked_blocks=replace(model.stacked_blocks, stacked=stacked, num_layers=1),
        )
        for export_name, actual in grugmoe_inference_state_dict(layer_model).items():
            name = export_name.replace("model.layers.0.", f"model.layers.{layer_index}.")
            name = name.replace(SHARED_EXPERT_LIST_KEY, SHARED_EXPERT_KEY)
            if name in checked:
                continue
            checked.add(name)
            if name not in source:
                errors.append(f"Missing source tensor: {name}")
                continue
            expected = source[name]
            if expected.shape != actual.shape:
                errors.append(f"Tensor shape mismatch: {name}")
                continue
            if not bool(jnp.array_equal(jax.sharding.reshard(expected, actual.sharding), actual)):
                errors.append(f"Tensor mismatch: {name}")
            records.append(dict(name=name, shape=list(actual.shape), dtype=str(actual.dtype)))
    errors.extend(f"Missing exported tensor: {name}" for name in sorted(set(source) - checked))
    if errors:
        raise ValueError("HF/native parity failed: " + "; ".join(errors))
    return records


def convert(hf_id: str, hf_revision: str, destination: str, config: GrugModelConfig) -> None:
    """Write ``destination/step-0``, restore it the way the checkpoint eval does, and require exact parity."""
    root = StoragePath.parse(destination)
    if root.exists() and root.ls():
        raise FileExistsError(f"Import requires a fresh destination: {root}")
    checkpoint_path = str(root / f"step-{CHECKPOINT_STEP}")
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    converter = SnowballConfig().hf_checkpoint_converter()
    with jax.set_mesh(mesh):
        source = converter.load_state_dict(RepoRef(hf_id, hf_revision), dtype=jnp.bfloat16)
        # The SFT exports name the single shared expert as a list of one; the base export names it bare.
        source = {name.replace(SHARED_EXPERT_LIST_KEY, SHARED_EXPERT_KEY): value for name, value in source.items()}
        model = import_weights(config, source)
        verify_weights(model, source)
        state = {"step": jax.sharding.reshard(jnp.array(CHECKPOINT_STEP, dtype=jnp.int32), P()), "params": model}
        manager = GlobalAsyncCheckpointManager(timeout_secs=SAVE_TIMEOUT)
        save_checkpoint(state, CHECKPOINT_STEP, checkpoint_path, manager=manager, is_temporary=False)
        manager.wait_until_finished()
        del model, state
        restored, step = load_checkpoint_params(config, checkpoint_path, mesh=mesh, mp=SNOWBALL_MIXED_PRECISION)
        if step != CHECKPOINT_STEP:
            raise ValueError(f"Restored step {step}, expected {CHECKPOINT_STEP}")
        records = verify_weights(restored, source)
    verification = dict(hf_id=hf_id, hf_revision=hf_revision, tensors=records, exact_parity=True)
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    logger.info("Verified all %d HF tensors after native checkpoint restore", len(records))


@click.command()
@click.option("--hf-id", required=True, help="Hugging Face repository of the Snowball export.")
@click.option("--hf-revision", required=True, help="Pinned commit of that repository.")
@click.option("--destination", required=True, help="Empty directory to receive step-0 and verification.json.")
def main(hf_id: str, hf_revision: str, destination: str) -> None:
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO)
    convert(hf_id, hf_revision, destination, SNOWBALL_67B_MODEL_CONFIG)


if __name__ == "__main__":
    main()
