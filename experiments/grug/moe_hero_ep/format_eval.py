# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned format panels and deferred training-source acquisition diagnostics."""

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec, get_mesh
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import MarinTokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.evaluation.eval_stats import wilson_interval
from pydantic import Field, model_validator

from experiments.grug.moe_hero_ep.model import _BATCH_AXES, Transformer, _seq_axis
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Prompt,
    Record,
    SampleResult,
    SamplingSpec,
    StopReason,
    digest,
)
from experiments.grug.moe_hero_ep.sft import SftConfig, SftParameters, sft_logits

EOT_ID = 128009
EOT_TEXT = "<|eot_id|>"
START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FORBIDDEN_CONTINUATION = ("<|start_header_id|>", "<|end_header_id|>", "<tool_response", "</tool_response>")


SOURCE_CONTEXT_LENGTH = 16384
EXTENDED_CONTEXT_LENGTH = 65536


class Interface(StrEnum):
    SIMPLE = "simple"
    REASONING_ON = "reasoning_on"
    REASONING_OFF = "reasoning_off"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    CONTEXT_16K = "context16k"
    CONTEXT_65K = "context65k"


class Serialization(StrEnum):
    MARIN = "marin"
    LEGACY = "legacy"


class PanelCase(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    interface: Interface
    messages: tuple[dict, ...] = Field(min_length=1)
    legacy_prompt: str = Field(min_length=1)
    template_kwargs: dict = Field(default_factory=dict)
    expected_tool_names: tuple[str, ...] = ()
    expected_observation_answer: str | None = None
    # Pin an actual long-token requirement rather than trusting a "16K" label.
    minimum_prompt_tokens: int = Field(default=1, ge=1, lt=EXTENDED_CONTEXT_LENGTH)
    seed: int = Field(ge=0)
    source: str = Field(min_length=1)
    expected: str | None = None

    @model_validator(mode="after")
    def interface_contract(self) -> Self:
        if any(key in self.template_kwargs for key in ("chat_template", "tokenize", "add_generation_prompt")):
            raise ValueError("Panel cases may not override the renderer or generation boundary")
        if self.interface == Interface.REASONING_ON and self.template_kwargs.get("enable_thinking") is not True:
            raise ValueError("Reasoning-on panel must request enable_thinking=true")
        if self.interface == Interface.REASONING_OFF and self.template_kwargs.get("enable_thinking") is not False:
            raise ValueError("Reasoning-off panel must request enable_thinking=false")
        if self.interface == Interface.TOOL_CALL and not self.expected_tool_names:
            raise ValueError("Tool-call cases must pin expected function names")
        if self.interface == Interface.OBSERVATION:
            if not self.expected_observation_answer or not any(message["role"] == "tool" for message in self.messages):
                raise ValueError("Observation cases need a tool response and an exact expected answer fragment")
        if self.interface == Interface.CONTEXT_16K and self.minimum_prompt_tokens < 16000:
            raise ValueError("16K check requires an actual prompt with at least 16000 tokens")
        if self.interface == Interface.CONTEXT_65K and self.minimum_prompt_tokens < 65000:
            raise ValueError("65K check requires an actual prompt with at least 65000 tokens")
        if self.messages[-1]["role"] == "assistant":
            raise ValueError("Generation panels must end at a user or tool turn")
        return self


class Panel(Record):
    release: str
    cases: tuple[PanelCase, ...] = Field(min_length=1)
    tokenizer: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    backend: str = Field(min_length=1)
    template_sha256: str = Field(default_factory=lambda: hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest())
    max_new_tokens: int = Field(gt=0, le=EXTENDED_CONTEXT_LENGTH)
    context_length: int = Field(default=SOURCE_CONTEXT_LENGTH)
    temperature: float = Field(ge=0, allow_inf_nan=False)

    @property
    def panel_id(self) -> str:
        return digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if self.context_length not in (SOURCE_CONTEXT_LENGTH, EXTENDED_CONTEXT_LENGTH):
            raise ValueError("Format panels support source context capacities of 16384 or 65536")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Panel case IDs must be unique")
        return self


def render_panel(panel: Panel, tokenizer: MarinTokenizer, serialization: Serialization) -> SamplingSpec:
    """Render the shared template or explicitly pinned legacy PT prompts."""
    if panel.template_sha256 != hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest():
        raise ValueError("The shared template changed since the panel was pinned")
    if serialization == Serialization.MARIN and tokenizer.encode(EOT_TEXT) != [EOT_ID]:
        raise ValueError("Prepared tokenizer does not match Hero's atomic Marin EOT")
    prompts = []
    for case in panel.cases:
        text = case.legacy_prompt
        if serialization == Serialization.MARIN:
            rendered = tokenizer.with_chat_template(MARIN_CHAT_TEMPLATE).apply_chat_template(
                list(case.messages),
                tokenize=False,
                add_generation_prompt=True,
                **case.template_kwargs,
            )
            assert isinstance(rendered, str)
            text = rendered
        ids = tokenizer.encode(text)
        if len(ids) < case.minimum_prompt_tokens or len(ids) >= panel.context_length:
            raise ValueError(f"Panel prompt does not meet its actual context limits: {case.id}")
        if case.expected is not None and len(ids) + len(tokenizer.encode(case.expected)) + 1 > panel.context_length:
            raise ValueError(f"Reference completion does not fit the actual context: {case.id}")
        prompts.append(Prompt(id=case.id, text=text, seed=case.seed, source_url=case.source, expected=case.expected))
    return SamplingSpec(
        release=f"{panel.release}-{serialization}-{panel.panel_id}",
        prompts=tuple(prompts),
        completions_per_prompt=1,
        tokenizer=panel.tokenizer,
        tokenizer_revision=panel.tokenizer_revision,
        temperature=panel.temperature,
        max_new_tokens=panel.max_new_tokens,
        context_length=panel.context_length,
    )


def completion_failures(
    case: PanelCase, text: str, token_ids: Sequence[int], stop_reason: StopReason
) -> tuple[str, ...]:
    """Classify raw continuation failures without discarding any generated text."""
    failures = []
    if stop_reason != StopReason.EOS or not token_ids or token_ids[-1] != EOT_ID:
        failures.append("termination")
    if text.count(EOT_TEXT) != 1 or not text.endswith(EOT_TEXT) or EOT_ID in token_ids[:-1]:
        failures.append("turn_boundary")
    if any(marker in text for marker in FORBIDDEN_CONTINUATION):
        failures.append("role_or_observation_continuation")
    content = text.removesuffix(EOT_TEXT)
    if case.interface == Interface.REASONING_ON:
        valid = (
            content.startswith(START_THINK)
            and content.count(START_THINK) == 1
            and content.count(END_THINK) == 1
            and bool(content.split(END_THINK)[-1].strip())
        )
        if not valid:
            failures.append("reasoning_mode")
    elif case.interface == Interface.REASONING_OFF and (START_THINK in content or END_THINK in content):
        failures.append("reasoning_mode")
    if case.interface == Interface.TOOL_CALL:
        calls = TOOL_CALL.findall(content)
        names = []
        parseable = len(calls) == content.count("<tool_call>") == content.count("</tool_call>")
        for call in calls:
            try:
                value = json.loads(call)
            except json.JSONDecodeError:
                parseable = False
                continue
            if not isinstance(value, dict) or not isinstance(value.get("arguments"), dict):
                parseable = False
            else:
                names.append(value.get("name"))
        if not parseable or tuple(names) != case.expected_tool_names:
            failures.append("tool_call")
    if case.interface == Interface.OBSERVATION:
        assert case.expected_observation_answer is not None
        if case.expected_observation_answer not in content:
            failures.append("observation_handling")
    return tuple(failures)


def grade_panel(panel: Panel, result: SampleResult, serialization: Serialization = Serialization.MARIN) -> dict:
    """Produce per-interface exact counts/95% Wilson intervals and retain every generation.

    This checks structural compliance. Content scoring and native-to-serving
    parity require separate measurements.
    """
    spec = result.request.spec
    if (
        spec.release != f"{panel.release}-{serialization}-{panel.panel_id}"
        or spec.tokenizer != panel.tokenizer
        or spec.tokenizer_revision != panel.tokenizer_revision
        or result.request.source_revision != panel.source_revision
        or spec.completions_per_prompt != 1
        or spec.context_length != panel.context_length
        or spec.temperature != panel.temperature
        or spec.max_new_tokens != panel.max_new_tokens
        or [prompt.id for prompt in spec.prompts] != [case.id for case in panel.cases]
    ):
        raise ValueError("Completed samples do not match the pinned panel and evaluation revision")
    if serialization == Serialization.MARIN and result.eos_token_id != EOT_ID:
        raise ValueError("Marin evaluation must stop on EOT while retaining special tokens")
    rows = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for case, completed in zip(panel.cases, result.completions, strict=True):
        sample = completed.samples[0]
        failures = (
            completion_failures(case, sample.text, sample.token_ids, sample.stop_reason)
            if serialization == Serialization.MARIN
            else ()
        )
        if len(completed.prompt_token_ids) < case.minimum_prompt_tokens:
            raise ValueError(f"Actual evaluated prompt misses the context check: {case.id}")
        row = {"id": case.id, "interface": case.interface.value, "failures": failures, "generation": sample.model_dump()}
        rows.append(row)
        groups[case.interface.value].append(row)
    metrics = {}
    if serialization == Serialization.MARIN:
        for interface, cases in groups.items():
            successes = sum(not row["failures"] for row in cases)
            interval = wilson_interval(successes, len(cases))
            metrics[interface] = {
                "successes": successes,
                "failures": len(cases) - successes,
                "total": len(cases),
                "observed_rate": successes / len(cases),
                "confidence_interval_95": [interval.low, interval.high],
                "interval_method": "Wilson score; operational fixed-panel rate is not a population guarantee",
            }
    report = {
        "panel_id": panel.panel_id,
        "serialization": serialization.value,
        "context_length": panel.context_length,
        "prompt_token_counts": {row.prompt_id: len(row.prompt_token_ids) for row in result.completions},
        "checkpoint": result.request.checkpoint.model_dump(),
        "backend": panel.backend,
        "model_config_digest": digest(result.request.model),
        "tokenizer_revision": panel.tokenizer_revision,
        "sample_id": result.request.sample_id,
        "metrics": metrics,
        "generations": rows,
    }
    report["report_digest"] = digest(report)
    return report


class AcquisitionCase(Record):
    source_id: str
    prefix_token_ids: tuple[int, ...] = Field(min_length=1)
    target_token_id: int = Field(ge=0)


class AcquisitionPanel(Record):
    source_artifact_digest: str
    tokenizer_revision: str
    cases: tuple[AcquisitionCase, ...] = Field(min_length=1)
    scope: str = "teacher-forced selected-token acquisition on fixed training-source examples"

    @property
    def panel_digest(self) -> str:
        return digest(self.model_dump())


def build_acquisition_panel(
    training_rows: Sequence[dict],
    tokenizer: MarinTokenizer,
    *,
    tokenizer_revision: str,
    source_artifact_digest: str,
    selected_token_ids: Sequence[int],
    max_examples: int,
) -> AcquisitionPanel:
    """Pin real supervised special-token targets, separately from heldout acceptance."""
    processor = ChatProcessor(tokenizer, chat_template=MARIN_CHAT_TEMPLATE, mask_user_turns=True)
    cases = []
    selected = set(selected_token_ids)
    for row in training_rows[:max_examples]:
        processed = processor([row])[0]
        ids, masks = processed["input_ids"], processed["assistant_masks"]
        for position in range(1, len(ids)):
            if masks[position] and int(ids[position]) in selected:
                cases.append(
                    AcquisitionCase(
                        source_id=row["id"],
                        prefix_token_ids=tuple(map(int, ids[:position])),
                        target_token_id=int(ids[position]),
                    )
                )
    if not cases:
        raise ValueError("Training-source diagnostic contains no supervised selected-token targets")
    return AcquisitionPanel(
        source_artifact_digest=source_artifact_digest, tokenizer_revision=tokenizer_revision, cases=tuple(cases)
    )


@eqx.filter_jit
def _acquisition_scores(frozen, params, token_ids, positions, targets, config):
    logits = sft_logits(frozen, params, token_ids, config, positions=positions)
    selected = logits.at[jnp.arange(logits.shape[0]), targets].get(out_sharding=PartitionSpec())
    normalizer = jax.sharding.reshard(jax.scipy.special.logsumexp(logits, axis=-1), PartitionSpec())
    nll = normalizer - selected
    return (
        jax.sharding.reshard(jnp.argmax(logits, axis=-1), PartitionSpec()),
        jax.sharding.reshard(nll, PartitionSpec()),
    )


@dataclass(frozen=True)
class AcquisitionDiagnostic:
    """Runtime callback; measures short teacher-forced acquisition outside update timing."""

    panel_json: str

    def __post_init__(self):
        panel = AcquisitionPanel.model_validate_json(self.panel_json)
        if self.panel_json != panel.model_dump_json():
            raise ValueError("AcquisitionDiagnostic requires canonical AcquisitionPanel.model_dump_json()")

    @classmethod
    def from_panel(cls, panel: AcquisitionPanel) -> Self:
        return cls(panel_json=panel.model_dump_json())

    def __call__(self, frozen: Transformer, params: SftParameters, config: SftConfig, step: int) -> dict:
        panel = AcquisitionPanel.model_validate_json(self.panel_json)
        mesh = get_mesh()
        if mesh.empty:
            raise ValueError("Acquisition diagnostic requires the same explicit model mesh as SFT")
        count = len(panel.cases)
        batch_size = math.ceil(count / jax.device_count()) * jax.device_count()
        context_axis = _seq_axis(mesh)
        sequence_multiple = int(mesh.shape[context_axis]) if context_axis else 1
        sequence_length = max(len(case.prefix_token_ids) for case in panel.cases)
        sequence_length = math.ceil(sequence_length / sequence_multiple) * sequence_multiple
        if sequence_length > frozen.config.max_seq_len:
            raise ValueError("Acquisition prefix exceeds the current model context")
        tokens = np.full((batch_size, sequence_length), panel.cases[0].target_token_id, dtype=np.int32)
        positions = np.empty(batch_size, dtype=np.int32)
        targets = np.empty(batch_size, dtype=np.int32)
        for index in range(batch_size):
            case = panel.cases[index % count]
            tokens[index, : len(case.prefix_token_ids)] = case.prefix_token_ids
            positions[index] = len(case.prefix_token_ids) - 1
            targets[index] = case.target_token_id
        token_sharding = NamedSharding(mesh, PartitionSpec(_BATCH_AXES, context_axis))
        row_sharding = NamedSharding(mesh, PartitionSpec(_BATCH_AXES))
        token_array = jax.make_array_from_callback(tokens.shape, token_sharding, lambda index: tokens[index])
        position_array = jax.make_array_from_callback(positions.shape, row_sharding, lambda index: positions[index])
        target_array = jax.make_array_from_callback(targets.shape, row_sharding, lambda index: targets[index])
        predicted, nll = _acquisition_scores(frozen, params, token_array, position_array, target_array, config)
        predicted, nll = np.asarray(predicted)[:count], np.asarray(nll)[:count]
        if not np.isfinite(nll).all():
            raise ValueError("Nonfinite acquisition likelihood; diagnose before continuing SFT")
        rows = [
            {
                "source_id": case.source_id,
                "target_id": case.target_token_id,
                "predicted_id": int(predicted[index]),
                "nll": float(nll[index]),
            }
            for index, case in enumerate(panel.cases)
        ]
        return {
            "successes": int(np.sum(predicted == targets[:count])),
            "cases": count,
            "mean_nll": float(np.mean(nll)),
            "panel_digest": panel.panel_digest,
            "rule": panel.scope,
            "report": {"step": step, "rows": rows},
        }
