# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure fixed held-out assistant targets on saved compact Hero checkpoints.

Preparation reads pinned chat JSONL on CPU. Evaluation runs in a separate job
after training, restoring the frozen base once for all compact checkpoints.
All forwards use one declared padded shape; this worker never takes gradients.
"""

import argparse
import gc
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.data.text.formats import ChatProcessor
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from pydantic import Field, model_validator
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.format_eval import EOT_ID
from experiments.grug.moe_hero_ep.model import _BATCH_AXES, GrugModelConfig, _seq_axis
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, Record, digest
from experiments.grug.moe_hero_ep.sft import (
    BaseReference,
    SftConfig,
    export_sft,
    initialize_from_checkpoint,
    merge_sft,
    restore_sft_checkpoint,
    sft_logits,
    sft_loss,
)
from experiments.grug.moe_hero_ep.train import _apply_hero_ep_runtime_defaults

logger = logging.getLogger(__name__)


class HeldoutCase(Record):
    id: str
    token_ids: tuple[int, ...] = Field(min_length=2, max_length=16384)
    next_token_weights: tuple[float, ...]

    @model_validator(mode="after")
    def assistant_targets(self):
        if (
            len(self.token_ids) != len(self.next_token_weights)
            or self.token_ids[-1] != EOT_ID
            or self.next_token_weights[-2] != 1
            or self.next_token_weights[-1] != 0
            or not all(weight in (0, 1) for weight in self.next_token_weights)
        ):
            raise ValueError("Heldout targets must align and include the final assistant EOT")
        return self


class HeldoutPanel(Record):
    source_manifest_digest: str
    heldout_sha256: str
    train_sha256: str
    tokenizer: str
    tokenizer_revision: str
    template_sha256: str
    cases: tuple[HeldoutCase, ...] = Field(min_length=1, max_length=1186)

    @model_validator(mode="after")
    def distinct_cases(self):
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Heldout conversations must be unique")
        return self


def prepare_panel(data_root: Path, destination: Path, max_examples: int) -> HeldoutPanel:
    """Render actual heldout conversations, retaining shifted assistant masks."""
    manifest = json.loads((data_root / "manifest.json").read_bytes())
    if (
        digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
        != manifest["manifest_digest"]
    ):
        raise ValueError("Prepared data manifest changed")
    if manifest["template_sha256"] != hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest():
        raise ValueError("Training and heldout evaluation must use the same chat template")
    raw = {split: (data_root / f"{split}.jsonl").read_bytes() for split in ("train", "heldout")}
    hashes = {split: hashlib.sha256(value).hexdigest() for split, value in raw.items()}
    if any(hashes[split] != manifest["outputs"][split]["sha256"] for split in raw):
        raise ValueError("Prepared split bytes changed")
    rows = {split: [json.loads(line) for line in value.splitlines()] for split, value in raw.items()}
    train_ids = {row["id"] for row in rows["train"]}
    if train_ids.intersection(row["id"] for row in rows["heldout"]):
        raise ValueError("Training and heldout conversation identities overlap")
    if max_examples < 1 or max_examples > len(rows["heldout"]):
        raise ValueError("Declare a nonempty bounded heldout selection")
    recipe = manifest["recipe"]
    tokenizer = load_tokenizer(f"{recipe['tokenizer_path']}@{recipe['tokenizer_revision']}")
    if digest(tokenizer.get_vocab()) != manifest["tokenizer_vocabulary_sha256"]:
        raise ValueError("Heldout tokenizer vocabulary differs from the prepared training data")
    if tokenizer.encode("<|eot_id|>") != [EOT_ID]:
        raise ValueError("Tokenizer must retain the pinned atomic EOT")
    processor = ChatProcessor(tokenizer, chat_template=MARIN_CHAT_TEMPLATE, mask_user_turns=True)
    selected = sorted(rows["heldout"], key=lambda row: row["id"])[:max_examples]
    cases = []
    for row in selected:
        processed = processor([row])[0]
        tokens = tuple(map(int, processed["input_ids"]))
        masks = tuple(map(int, processed["assistant_masks"]))
        if masks[-1] != 1:
            raise ValueError("Final assistant EOT must be supervised")
        cases.append(HeldoutCase(id=row["id"], token_ids=tokens, next_token_weights=(*masks[1:], 0.0)))
    panel = HeldoutPanel(
        source_manifest_digest=manifest["manifest_digest"],
        heldout_sha256=hashes["heldout"],
        train_sha256=hashes["train"],
        tokenizer=recipe["tokenizer_path"],
        tokenizer_revision=recipe["tokenizer_revision"],
        template_sha256=hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest(),
        cases=tuple(cases),
    )
    with destination.open("x") as stream:
        stream.write(panel.model_dump_json(indent=2) + "\n")
    return panel


class HeldoutRequest(Record):
    reference: BaseReference
    sft: SftConfig
    model: GrugModelConfig
    checkpoints: tuple[Checkpoint, ...] = Field(min_length=3, max_length=3)
    panel_path: str
    panel_sha256: str
    source_revision: str
    destination: str
    batch_size: int
    physical_padding: int
    expert_axis_size: int
    context_axis_size: int
    replica_axis_size: int


def read_panel(request: HeldoutRequest) -> HeldoutPanel:
    if {checkpoint.step for checkpoint in request.checkpoints} != {64, 96, 128}:
        raise ValueError("Selection requires the pinned 64/96/128 compact checkpoints")
    raw = StoragePath(request.panel_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != request.panel_sha256:
        raise ValueError("Heldout panel bytes changed")
    panel = HeldoutPanel.model_validate_json(raw)
    if (panel.tokenizer, panel.tokenizer_revision) != (
        request.reference.tokenizer,
        request.reference.tokenizer_revision,
    ):
        raise ValueError("Heldout tokenizer differs from the frozen base")
    if panel.template_sha256 != hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest():
        raise ValueError("Heldout chat template changed")
    if not max(len(case.token_ids) for case in panel.cases) <= request.physical_padding <= request.model.max_seq_len:
        raise ValueError("Heldout sequences must fit without truncation")
    if request.batch_size < 1 or request.physical_padding % request.context_axis_size:
        raise ValueError("Evaluation shape does not fit the declared mesh")
    return panel


@eqx.filter_jit
def teacher_forced_metrics(frozen, params, tokens, weights, segments, positions, config):
    mask = AttentionMask.causal().with_segment_ids(segments)
    assistant_loss, routing = sft_loss(params, frozen, tokens, weights, config, mask)
    routing = {
        name: routing[name]
        for name in (
            "train/router/routing_entropy_mean",
            "train/router/router_z_loss",
            "train/router/routing_counts_per_layer",
            "train/router/capacity_overflow_rate_mean",
        )
    }
    logits = sft_logits(frozen, params, tokens, config, mask, positions=positions)
    eot_nll = jax.scipy.special.logsumexp(logits, axis=-1) - logits[:, EOT_ID]
    greedy = jnp.argmax(logits, axis=-1)
    return jax.tree.map(lambda value: jax.sharding.reshard(value, P()), (assistant_loss, routing, eot_nll, greedy))


def heldout_batches(panel: HeldoutPanel, batch_size: int, physical_padding: int):
    """Yield full-shape batches with masked filler and causal target alignment."""
    for start in range(0, len(panel.cases), batch_size):
        cases = panel.cases[start : start + batch_size]
        tokens = np.zeros((batch_size, physical_padding), np.int32)
        weights = np.zeros(tokens.shape, np.float32)
        segments = np.full(tokens.shape, -1, np.int32)
        positions = np.zeros(batch_size, np.int32)
        for index, case in enumerate(cases):
            count = len(case.token_ids)
            tokens[index, :count] = case.token_ids
            weights[index, :count] = case.next_token_weights
            segments[index, :count] = 0
            positions[index] = count - 2
        yield cases, tokens, weights, segments, positions


def measure_compact(frozen, state, panel: HeldoutPanel, request: HeldoutRequest, mesh) -> dict:
    token_sharding = NamedSharding(mesh, P(_BATCH_AXES, _seq_axis(mesh)))
    row_sharding = NamedSharding(mesh, P(_BATCH_AXES))
    total_loss, total_weight = 0.0, 0.0
    rows, batches = [], []
    for cases, tokens, weights, segments, positions in heldout_batches(
        panel, request.batch_size, request.physical_padding
    ):
        arrays = [
            jax.make_array_from_callback(value.shape, token_sharding, lambda index, value=value: value[index])
            for value in (tokens, weights, segments)
        ]
        last = jax.make_array_from_callback(
            positions.shape, row_sharding, lambda index, positions=positions: positions[index]
        )
        loss, routing, eot_nll, greedy = jax.device_get(
            teacher_forced_metrics(frozen, state.params, *arrays, last, request.sft)
        )
        if not np.isfinite(loss) or not np.all(np.isfinite(eot_nll[: len(cases)])):
            raise ValueError("Nonfinite heldout likelihood")
        weight = float(weights.sum(dtype=np.float64))
        total_loss += float(loss) * weight
        total_weight += weight
        rows.extend(
            {"id": case.id, "eot_nll": float(eot_nll[index]), "eot_top1": int(greedy[index]) == EOT_ID}
            for index, case in enumerate(cases)
        )
        batches.append(
            {
                "assistant_loss": float(loss),
                "assistant_targets": int(weight),
                "routing": jax.tree.map(lambda value: np.asarray(value).tolist(), routing),
            }
        )
    return {
        "step": int(state.step),
        "assistant_nll": total_loss / total_weight,
        "assistant_targets": int(total_weight),
        "eot_mean_nll": float(np.mean([row["eot_nll"] for row in rows])),
        "eot_top1": sum(row["eot_top1"] for row in rows),
        "conversations": len(rows),
        "rows": rows,
        "batches": batches,
        "scope": "Fixed heldout teacher forcing; EOT predictions are not generated turn-termination measurements",
    }


def initialize_worker(request: HeldoutRequest):
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        processes_per_task=4,
        moe_implementation=request.model.moe_implementation,
        remat_mode=request.model.remat_mode,
    )
    DistributedConfig().initialize()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    install(cutlass_kernel_cache())
    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise ValueError("Evaluation requires one GPU per process")
    mesh = compact_grug_mesh(
        expert_axis_size=request.expert_axis_size,
        replica_axis_size=request.replica_axis_size,
        context_axis_size=request.context_axis_size,
    )
    if request.batch_size % (jax.device_count() // request.context_axis_size):
        raise ValueError("Evaluation batch must divide across the requested ranks")
    return mesh


def restored_checkpoint(checkpoint: Checkpoint, template, request: HeldoutRequest, mesh):
    metadata = json.loads((StoragePath(checkpoint.uri) / "metadata.json").read_bytes())
    if digest(metadata) != checkpoint.metadata_digest or metadata.get("step") != checkpoint.step:
        raise ValueError("Compact checkpoint identity changed")
    state = restore_sft_checkpoint(checkpoint.uri, template, request.reference, request.sft, mesh)
    if int(state.step) != checkpoint.step:
        raise ValueError("Compact state differs from the selected step")
    return state


def select_checkpoint(measurements: list[dict]) -> int:
    """Choose minimum assistant NLL without reducing EOT top-1 from update 64."""
    baseline = next(row for row in measurements if row["step"] == 64)
    eligible = [row for row in measurements if row["eot_top1"] >= baseline["eot_top1"]]
    return min(eligible, key=lambda row: (row["assistant_nll"], row["step"]))["step"]


def run_evaluation(request: HeldoutRequest) -> None:
    panel = read_panel(request)
    root = StoragePath(request.destination)
    if root.exists():
        raise FileExistsError("Choose a fresh heldout destination")
    mesh = initialize_worker(request)
    with jax.set_mesh(mesh):
        frozen, template = initialize_from_checkpoint(request.reference, request.model, mesh, request.sft)
        reports = []
        for checkpoint in request.checkpoints:
            state = restored_checkpoint(checkpoint, template, request, mesh)
            measured = measure_compact(frozen, state, panel, request, mesh)
            reports.append({"checkpoint": checkpoint.model_dump(), **measured})
            logger.info(
                "Heldout step %d: assistant NLL %.6f, EOT NLL %.6f",
                state.step,
                measured["assistant_nll"],
                measured["eot_mean_nll"],
            )
    report = {
        "request": request.model_dump(mode="json"),
        "completed_at": datetime.now(UTC).isoformat(),
        "mesh": dict(mesh.shape),
        "measurements": reports,
        "selected_step": select_checkpoint(reports),
        "selection_rule": (
            "Among updates 64/96/128 with EOT top-1 at least update 64, choose minimum heldout assistant NLL; "
            "an exact tie selects the earlier update. "
            "This fixed panel is used for selection, not an unbiased final test."
        ),
    }
    report["report_digest"] = digest(report)
    if jax.process_index() == 0:
        conditional_object(str(root / "report.json")).write(
            (json.dumps(report, indent=2) + "\n").encode(), expected_version=None
        )


@eqx.filter_jit
def merged_probe_logits(model, tokens, segments, positions):
    hidden, _ = model(tokens, AttentionMask.causal().with_segment_ids(segments))
    selected = hidden.at[jnp.arange(tokens.shape[0]), positions].get(out_sharding=P())
    logits = jnp.einsum("bh,hv->bv", selected, model.output_proj, preferred_element_type=jnp.float32)
    return jax.sharding.reshard(logits, P())


def compare_export_logits(compact: np.ndarray, merged: np.ndarray) -> dict:
    """Summarize observed full-vocabulary differences without an acceptance tolerance."""
    if compact.shape != merged.shape or compact.ndim != 2 or compact.shape[0] == 0:
        raise ValueError("Merge probes must have matching nonempty row/vocabulary shapes")
    if not np.all(np.isfinite(compact)) or not np.all(np.isfinite(merged)):
        return {"finite": False, "probes": compact.shape[0]}
    compact, merged = compact.astype(np.float64), merged.astype(np.float64)
    compact_shifted = compact - compact.max(axis=-1, keepdims=True)
    merged_shifted = merged - merged.max(axis=-1, keepdims=True)
    compact_logprobs = compact_shifted - np.log(np.exp(compact_shifted).sum(axis=-1, keepdims=True))
    merged_logprobs = merged_shifted - np.log(np.exp(merged_shifted).sum(axis=-1, keepdims=True))
    logit_delta = np.abs(compact - merged)
    logprob_delta = np.abs(compact_logprobs - merged_logprobs)
    probability_delta = np.abs(np.exp(compact_logprobs) - np.exp(merged_logprobs))
    return {
        "finite": True,
        "probes": compact.shape[0],
        "vocabulary_size": compact.shape[1],
        "max_absolute_logit_delta": float(logit_delta.max()),
        "mean_absolute_logit_delta": float(logit_delta.mean()),
        "max_absolute_logprob_delta": float(logprob_delta.max()),
        "mean_absolute_logprob_delta": float(logprob_delta.mean()),
        "max_absolute_probability_delta": float(probability_delta.max()),
        "mean_absolute_probability_delta": float(probability_delta.mean()),
        "greedy_disagreements": int(np.count_nonzero(compact.argmax(axis=-1) != merged.argmax(axis=-1))),
    }


def measure_export(frozen, state, panel: HeldoutPanel, request: HeldoutRequest, mesh) -> dict:
    """Compare one fixed heldout batch before export and release the extra merged model."""
    cases, tokens, _, segments, positions = next(heldout_batches(panel, request.batch_size, request.physical_padding))
    token_sharding = NamedSharding(mesh, P(_BATCH_AXES, _seq_axis(mesh)))
    row_sharding = NamedSharding(mesh, P(_BATCH_AXES))
    ids, document_ids = [
        jax.make_array_from_callback(value.shape, token_sharding, lambda index, value=value: value[index])
        for value in (tokens, segments)
    ]
    last = jax.make_array_from_callback(positions.shape, row_sharding, lambda index: positions[index])
    compact = sft_logits(
        frozen,
        state.params,
        ids,
        request.sft,
        AttentionMask.causal().with_segment_ids(document_ids),
        positions=last,
    )
    merged = merge_sft(frozen, state.params, request.sft)
    ordinary = merged_probe_logits(merged, ids, document_ids, last)
    jax.block_until_ready((compact, ordinary))
    compact_host, ordinary_host = jax.device_get((compact, ordinary))
    del merged, compact, ordinary
    gc.collect()
    return {
        "panel_sha256": request.panel_sha256,
        "case_ids": [case.id for case in cases],
        "prediction_positions": positions[: len(cases)].tolist(),
        "measurements": compare_export_logits(
            np.asarray(compact_host)[: len(cases)], np.asarray(ordinary_host)[: len(cases)]
        ),
        "scope": (
            "One final-EOT prediction per conversation in the first heldout batch, comparing all vocabulary logits. "
            "Observed BF16 merge differences have no declared acceptance tolerance. "
            "This does not qualify serving parity. "
            "An extra merged model is resident for this probe, then released before the export merge."
        ),
    }


def export_selected(request: HeldoutRequest, step: int, destination: str) -> None:
    """Export exactly one explicitly selected compact checkpoint as ordinary weights."""
    report = json.loads((StoragePath(request.destination) / "report.json").read_bytes())
    if (
        report["request"] != request.model_dump(mode="json")
        or digest({key: value for key, value in report.items() if key != "report_digest"}) != report["report_digest"]
        or step != report["selected_step"]
    ):
        raise ValueError("Selection report does not match this evaluation request")
    selected = [checkpoint for checkpoint in request.checkpoints if checkpoint.step == step]
    if len(selected) != 1 or not any(
        measurement["checkpoint"] == selected[0].model_dump() for measurement in report["measurements"]
    ):
        raise ValueError("Select exactly one evaluated compact checkpoint")
    panel = read_panel(request)
    mesh = initialize_worker(request)
    with jax.set_mesh(mesh):
        frozen, template = initialize_from_checkpoint(request.reference, request.model, mesh, request.sft)
        state = restored_checkpoint(selected[0], template, request, mesh)
        parity = {
            "checkpoint": selected[0].model_dump(),
            "source_revision": request.source_revision,
            **measure_export(frozen, state, panel, request, mesh),
        }
        parity["report_digest"] = digest(parity)
        if jax.process_index() == 0:
            conditional_object(str(StoragePath(destination) / "merge-parity.json")).write(
                (json.dumps(parity, indent=2) + "\n").encode(), expected_version=None
            )
        if not parity["measurements"]["finite"]:
            raise ValueError("Nonfinite compact/merged logits; ordinary export was not written")
        export_sft(destination, frozen, state, request.reference, request.sft)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("data_root", type=Path)
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("--max-examples", type=int, default=256)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("request", type=Path)
    export = commands.add_parser("export")
    export.add_argument("request", type=Path)
    export.add_argument("--step", type=int, required=True)
    export.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare_panel(args.data_root, args.destination, args.max_examples)
        return
    configure_coreweave_s3()
    request = HeldoutRequest.model_validate_json(args.request.read_bytes())
    if args.operation == "evaluate":
        run_evaluation(request)
        return
    export_selected(request, args.step, args.destination)


if __name__ == "__main__":
    main()
