# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind T3 sampling to unchanged T2 inputs and the CD16 training/precision/export loop."""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass

import draccus
import numpy as np
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.data.text.datasets import LmDataConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmExample
from levanter.tracker import NoopConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.interp.snowball_identity.multihost import gather_global, gather_host_scalar
from experiments.post_training.identity_context_distill.lr0_probe import SOURCE_FILE_HASHES, check_pod_storage
from experiments.post_training.identity_context_distill.runtime_recipe import (
    configure_precision_runtime,
    require_xla_precision,
    xla_precision_in_process,
)
from experiments.post_training.identity_context_distill.save_audit import audit_trained_export
from experiments.post_training.identity_context_distill.sft import BASE_MODEL, identity_sft_config
from experiments.post_training.identity_context_distill.sft_launch import (
    SftAudits,
    read_remote,
    run_identity_sft,
    sft_runtime_receipt,
)
from experiments.post_training.identity_t2.prepare import (
    BATCH_SIZE,
    SEED,
    SEQ_LEN,
    CapturedChatFormat,
    array_sha,
    check_validity,
    sha,
    summarize,
)

LOGICAL_PACKET_SHA = "4f320bf5226777bcf173c0e7c52f14bc02e28f329ee12d9ec0f53cc0b6baee7e"
UPDATES = 16


@dataclass(frozen=True)
class FrozenInputs:
    root: str
    freeze_sha256: str
    freeze: dict
    data_config: dict
    ordered: list[dict]
    native: dict
    fingerprints: dict
    validity: dict


def checked_manifest(root: str, freeze_sha256: str) -> dict:
    raw = read_remote(root + "/freeze.json")
    if sha(raw) != freeze_sha256:
        raise ValueError("Regional freeze manifest differs from reviewed bytes")
    freeze = json.loads(raw)
    if freeze["root"] != root or freeze["logical_packet_sha256"] != LOGICAL_PACKET_SHA:
        raise ValueError("Regional freeze is not the logically cleared T3 dataset")
    return freeze


def load_inputs(root: str, freeze_sha256: str) -> FrozenInputs:
    """Re-read every small staged object; callers synchronize read failures across ranks."""
    freeze = checked_manifest(root, freeze_sha256)
    contents = {}
    for name, info in freeze["files"].items():
        raw = read_remote(root + "/" + name)
        if len(raw) != info["bytes"] or sha(raw) != info["sha256"]:
            raise ValueError(f"Regional data object changed: {name}")
        contents[name] = raw
    native = json.loads(contents["native-row-fingerprints.json"])
    fingerprints = {(row["tokens_sha256"], row["target_mask_sha256"]): identifier for identifier, row in native.items()}
    validity = {row["id"]: row for row in json.loads(contents["all-native-validity.json"])}
    ordered = [json.loads(line) for line in contents["occurrences-128.jsonl"].splitlines()]
    assert len(native) == len(fingerprints) == len(validity) == 1089 and len(ordered) == 128
    assert summarize(ordered) == freeze["first128"]
    assert freeze["first128"]["identity_token_share"] <= 0.08
    return FrozenInputs(
        root, freeze_sha256, freeze, json.loads(contents["data-config.json"]), ordered, native, fingerprints, validity
    )


def verify_arrays(tokens, weights, query, keys, update: int, frozen: FrozenInputs) -> dict:
    """Verify the actual global arrays, including explicit native dispatch segments."""
    if tokens.shape != (BATCH_SIZE, SEQ_LEN) or not tokens.shape == weights.shape == query.shape == keys.shape:
        raise ValueError("Expected eight full native examples with query/key segments")
    if not all(np.issubdtype(value.dtype, np.integer) for value in (tokens, query, keys)) or not np.all(
        (weights == 0) | (weights == 1)
    ):
        raise ValueError("Expected integer raw IDs and binary native loss masks")
    occurrences = []
    checks = []
    for offset, (row, mask, qseg, kseg) in enumerate(zip(tokens, weights, query, keys, strict=True)):
        identifier = frozen.fingerprints.get((array_sha(row), array_sha(mask)))
        if identifier is None:
            raise ValueError("Consumed token IDs or target masks differ from frozen native rows")
        position = update * BATCH_SIZE + offset
        if identifier != frozen.ordered[position]["id"]:
            raise ValueError("Consumed example differs from the frozen first128 order")
        meta = frozen.native[identifier]
        if int(mask.sum()) != meta["supervised_tokens"]:
            raise ValueError("Consumed native target token count differs")
        validity = frozen.validity[identifier]
        checked = check_validity(qseg, kseg, {**meta, "valid_input_length": validity["valid_input_length"]})
        if array_sha(qseg) != validity["query_segments_sha256"] or array_sha(kseg) != validity["key_segments_sha256"]:
            raise ValueError("Consumed native query/key segments differ from regional readback")
        occurrences.append({"position": position, "update": update, **meta})
        checks.append(checked)
    summary = summarize(occurrences)
    if summary != frozen.freeze["per_update"][update]:
        raise ValueError("Consumed per-update class/token accounting differs")
    return {
        "update": update,
        "ids": [row["id"] for row in occurrences],
        **summary,
        "validity": checks,
        "coefficient_mass_not_gradient_share": True,
    }


def collective_failure(error: Exception | None) -> None:
    if np.any(gather_host_scalar(int(error is not None))):
        raise ValueError("T3 dataset audit failed on at least one rank") from error


def checked_batch(batch: LmExample, update: int, frozen: FrozenInputs) -> dict:
    error = None
    arrays = None
    try:
        if not batch.attn_mask.is_causal:
            raise ValueError("T3 training requires the unchanged native causal attention mask")
        query, keys = batch.attn_mask.segment_ids
        arrays = (batch.tokens.array, batch.loss_weight.array, query.array, keys.array)
    except Exception as exc:
        error = exc
    # All ranks must pass local structure checks before any array collective.
    collective_failure(error)
    assert arrays is not None
    gathered = tuple(gather_global(value) for value in arrays)
    error = None
    receipt = None
    try:
        receipt = verify_arrays(*gathered, update, frozen)
    except Exception as exc:
        error = exc
    collective_failure(error)
    assert receipt is not None
    return receipt


def audit_iterator(
    batches: Iterator[LmExample], config: TrainLmConfig, frozen: FrozenInputs, run_root: str
) -> tuple[Iterator[LmExample], dict]:
    error = None
    try:
        if config.data_seed != SEED or config.train_seq_len != SEQ_LEN:
            raise ValueError("T3 requires unchanged seed and context length")
        if config.trainer.train_batch_size != BATCH_SIZE or config.trainer.microbatch_size != BATCH_SIZE:
            raise ValueError("T3 requires CD16's global and microbatch size of eight")
        if draccus.encode(config.data) != frozen.data_config:
            raise ValueError("Resolved data differs from the regional frozen config")
    except Exception as exc:
        error = exc
    collective_failure(error)
    buffered, reports = [], []
    for update in range(UPDATES):
        error = None
        batch = None
        try:
            batch = next(batches)
        except Exception as exc:
            error = exc
        collective_failure(error)
        assert batch is not None
        reports.append(checked_batch(batch, update, frozen))
        buffered.append(batch)
    ids = [identifier for report in reports for identifier in report["ids"]]
    pooled = summarize([frozen.native[identifier] for identifier in ids])
    error = None
    if pooled != frozen.freeze["first128"]:
        error = ValueError("Actual trainer first128 token accounting differs")
    collective_failure(error)
    report = {
        "freeze_root": frozen.root,
        "freeze_sha256": frozen.freeze_sha256,
        "logical_packet_sha256": LOGICAL_PACKET_SHA,
        "same_iterator_batches_buffered": UPDATES,
        "completed_updates": 0,
        "first128": pooled,
        "updates": reports,
        "explicit_query_key_segment_checks": 128,
        "source_history_supervised_in_new_rows": False,
        "sampling_not_loss_weights": True,
    }
    run_on_export_writer(lambda: write_export_json(StoragePath(run_root) / "audits/dataset-step-0.json", report))
    return itertools.chain(buffered, batches), report


def t3_config(
    run_root: str,
    run_id: str,
    config_bytes: bytes,
    generation_bytes: bytes,
    data_config_bytes: bytes,
    runtime_receipt: dict,
) -> TrainLmConfig:
    data = draccus.decode(LmDataConfig, json.loads(data_config_bytes))
    expected_order = ["identity", "preservation", "identity_v2", "tools", "third_person"]
    if list(data.components) != expected_order or data.train_weights != {
        "identity": 264.0,
        "preservation": 547.0,
        "identity_v2": 504.0,
        "tools": 468.0,
        "third_person": 324.0,
    }:
        raise ValueError("T3 component order or sampling weights differ")
    if data.auto_build_caches or data.permutation_type != "feistel" or data.mixture_block_size != 2048:
        raise ValueError("T3 requires frozen caches and CD16 sampling")
    for name, component in data.components.items():
        if component.pack is not False or component.format.pack is not False:
            raise ValueError("T3 rows must remain unpacked native chat examples")
        if component.format.slice_strategy != "raise" or not component.format.mask_user_turns:
            raise ValueError("T3 cannot crop or supervise source user turns")
        if name in ("identity_v2", "tools", "third_person") and not isinstance(component.format, CapturedChatFormat):
            raise ValueError("New T3 rows must use the native captured-target format")
    identity = data.components["identity"]
    base = identity_sft_config(
        dataset_uri=identity.source.train_urls[0],
        cache_uri=data.cache_dir,
        output_uri=run_root,
        run_id=run_id,
        seed=SEED,
        source_config=json.loads(config_bytes),
        source_generation_config=json.loads(generation_bytes),
        hosts=4,
        moe_capacity_factor=4.0,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation=runtime_receipt["runtime_model_config"]["attention_implementation"],
        stop_after_updates=UPDATES,
    )
    return dataclasses.replace(base, data=data)


def load_audits(root: str, freeze_sha256: str, run_root: str, lr0_source: dict) -> SftAudits:
    error = None
    frozen = None
    try:
        frozen = load_inputs(root, freeze_sha256)
    except Exception as exc:
        error = exc
    collective_failure(error)
    assert frozen is not None
    return SftAudits(
        audit_iterator=lambda batches, config: audit_iterator(batches, config, frozen, run_root),
        check_batch=lambda batch, update: checked_batch(batch, update, frozen),
        save_audit=lambda state, export_path: audit_trained_export(
            state,
            export_path,
            source_root=BASE_MODEL,
            source_receipt_path=lr0_source["path"],
            source_receipt_sha256=lr0_source["sha256"],
            expected_hashes=SOURCE_FILE_HASHES,
        ),
    )


def bound_worker(
    run_root: str,
    run_id: str,
    receipt_bytes: bytes,
    receipt_sha256: str,
    root: str,
    freeze_sha256: str,
    lr0_source: dict,
    source_objects: dict,
) -> None:
    """Use the original checked training loop; replace only its explicit data/audit inputs."""
    runtime_receipt = sft_runtime_receipt(receipt_bytes, receipt_sha256)
    require_xla_precision(runtime_receipt["xla_precision"])
    check_pod_storage()
    freeze = checked_manifest(root, freeze_sha256)
    data_bytes = read_remote(root + "/data-config.json")
    if sha(data_bytes) != freeze["files"]["data-config.json"]["sha256"]:
        raise ValueError("Worker data config differs from regional readback")
    config_bytes = read_remote(BASE_MODEL + "config.json")
    generation_bytes = read_remote(BASE_MODEL + "generation_config.json")
    base = t3_config(run_root, run_id, config_bytes, generation_bytes, data_bytes, runtime_receipt)
    reference = runtime_receipt["routing_reference"]
    config, observed = configure_precision_runtime(
        base,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(reference["path"]).read_bytes(),
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
        xla_precision=xla_precision_in_process(),
    )
    if observed != runtime_receipt or config.data != base.data:
        raise ValueError("T3 changed the reviewed CD16 precision runtime or data")
    config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, tracker=NoopConfig()))
    result = run_identity_sft(
        config,
        runtime_receipt=runtime_receipt,
        receipt_sha256=receipt_sha256,
        lr0_source=lr0_source,
        source_objects=source_objects,
        run_root=run_root,
        load_audits=lambda: load_audits(root, freeze_sha256, run_root, lr0_source),
    )
    run_on_export_writer(lambda: write_export_json(StoragePath(run_root) / "sft-result.json", result))
