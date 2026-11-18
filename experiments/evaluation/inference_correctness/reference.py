# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate Snowball reference scores separately from routine endpoint checks."""

import dataclasses
import hashlib
import json
import logging
import time

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax import Axis
from jax.sharding import PartitionSpec as P
from levanter.compat.hf_checkpoints import RepoRef
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.testing.inference.goldens import (
    Artifact,
    ModelIdentity,
    ReferenceCase,
    ReferenceManifest,
    ScoreRow,
    canonical_bytes,
    history_digest,
    verified_bytes,
)
from marin.testing.inference.snowball_backend_parity_jobs import _with_attention_implementation
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)
TOP_K = 64


def score_snowball_positions(model: SnowballLMHeadModel, tokens, positions):
    """Project selected causal positions without materializing sequence-wide logits."""
    hidden = model.activations(tokens).rearrange(("batch", "position", "embed")).array
    selected = hidden.at[jnp.arange(hidden.shape[0])[:, None], positions].get(out_sharding=P("data"))

    def project(state):
        logits = jnp.einsum("bh,hv->bv", state, model.transformer.output_proj, out_sharding=P("data"))
        values = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return jax.lax.top_k(values, min(TOP_K, model.Vocab.size))

    values, token_ids = jax.lax.map(project, jnp.swapaxes(selected, 0, 1))
    return jnp.swapaxes(values, 0, 1), jnp.swapaxes(token_ids, 0, 1)


def save_immutable(uri: str, raw: bytes) -> Artifact:
    """Publish or reuse identical bytes; callers must use a single writer."""
    filesystem, path = filesystem_for(uri)
    if filesystem.exists(path):
        with filesystem.open(path, "rb") as stream:
            if stream.read() != raw:
                raise ValueError(f"Reference already exists with different bytes: {uri}")
    else:
        with filesystem.open(path, "wb") as stream:
            stream.write(raw)
    return Artifact(uri=uri, sha256=hashlib.sha256(raw).hexdigest())


def generate_snowball_reference(
    identity: dict, cases_artifact: dict, training_config: dict, output: str, source_commit: str
) -> None:
    """Score the pinned HF weights with the model used by their identity SFT."""
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO)
    started = time.time()
    model_identity = ModelIdentity.model_validate(identity)
    artifact = Artifact.model_validate(cases_artifact)
    cases = tuple(ReferenceCase.model_validate(case) for case in json.loads(verified_bytes(artifact)))
    raw_config = dict(training_config)
    raw_config.pop("type")
    config = draccus.decode(SnowballConfig, raw_config)
    config = dataclasses.replace(config, moe_implementation="sonic", attention_implementation="gpu_fa4_cute")
    # The published HF metadata says "production". Its own SFT/export records specify this
    # Snowball class. Supplying that recorded config avoids silently inferring another model.
    reference = RepoRef(model_identity.weights, model_identity.revision)
    converter = config.hf_checkpoint_converter(ref_checkpoint=str(reference))
    trainer = TrainerConfig(
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        mesh=MeshConfig(axes={"replica": 1, "data": jax.device_count(), "model": 1, "expert": 1, "context": 1}),
        use_explicit_mesh_axes=True,
    )
    rows = []
    timings = []
    with trainer.use_device_mesh():
        model = converter.load_pretrained(
            SnowballLMHeadModel,
            ref=reference,
            config=config,
            dtype=jnp.bfloat16,
            axis_mapping=trainer.parameter_axis_mapping,
        )
        reference_model = _with_attention_implementation(model, "reference")

        score_positions = hax.named_jit(score_snowball_positions, axis_resources=trainer.compute_axis_mapping)
        batch_size = jax.device_count()
        ordered = sorted(cases, key=lambda case: len(case.tokens))
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            length = next(
                value for value in (256, 1024, 4096, 16384, 32768) if value >= max(len(c.tokens) for c in batch)
            )
            width = max(len(case.positions) for case in batch)
            tokens = np.full((batch_size, length), model_identity.eos_token_id, dtype=np.int32)
            positions = np.zeros((batch_size, width), dtype=np.int32)
            for index in range(batch_size):
                case = batch[index % len(batch)]
                tokens[index, : len(case.tokens)] = case.tokens
                positions[index, : len(case.positions)] = case.positions
            scoring_model = reference_model if length <= 4096 else model
            before = time.time()
            values, ids = score_positions(
                scoring_model,
                hax.named(jnp.asarray(tokens), (Axis("batch", batch_size), Axis("position", length))),
                jnp.asarray(positions),
            )
            values, ids = np.asarray(values), np.asarray(ids)
            for index, case in enumerate(batch):
                for column, position in enumerate(case.positions):
                    rows.append(
                        ScoreRow(
                            case_id=case.id,
                            position=position,
                            token_ids=tuple(ids[index, column].tolist()),
                            logprobs=tuple(values[index, column].tolist()),
                            history_sha256=history_digest(case.tokens[: position + 1]),
                        )
                    )
            timings.append(
                {"cases": [case.id for case in batch], "padded_length": length, "seconds": time.time() - before}
            )
            logger.info("Reference batch: %s", timings[-1])
    scores = save_immutable(
        prefix_join(output, "scores.json"), canonical_bytes([row.model_dump(mode="json") for row in rows])
    )
    manifest = ReferenceManifest(
        model=model_identity,
        suite_id=artifact.sha256,
        cases=artifact,
        scores=scores,
        reference_source="lib/levanter/src/levanter/models/snowball.py",
        reference_commit=source_commit,
        reference_config=draccus.encode(config),
        lineage={
            "training_config": training_config,
            "weights": str(reference),
            "compute_policy": "BF16 weights and projection; FP32 log-softmax",
            "devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
            "timings": timings,
            "elapsed_seconds": time.time() - started,
        },
    )
    saved = save_immutable(prefix_join(output, "manifest.json"), canonical_bytes(manifest.model_dump(mode="json")))
    logger.info("Reference manifest: %s", saved.model_dump_json())
