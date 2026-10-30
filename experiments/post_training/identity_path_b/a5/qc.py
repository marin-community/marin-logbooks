# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen Held-profile prescreens; passes still need semantic pool admission."""

import re
from collections import Counter

from experiments.post_training.identity_path_b.data import LOW_WEIGHT_IDS, SPECIAL_TOKEN_IDS
from experiments.post_training.identity_path_b.reasoning import (
    END_THINK,
    EOT,
    START_THINK,
    foreign_claim,
    identity_qc,
    omitted_context,
)
from experiments.post_training.identity_path_b.replay_capture import SOURCE_MODEL
from experiments.post_training.identity_path_b.trace_semantics import (
    ARCHITECTURE,
    DEPLOYMENT,
    LICENSE,
    LINEAGE,
    LINEAGE_V2,
    ORGANIZATION_V2,
    REFERENCE,
    ROLE_EXCLUSION_V2,
    TRAINING_CONTRIBUTOR,
    UI_CAPABILITY_V2,
    UNKNOWN,
    UNSUPPORTED_EXCLUSION,
)
from experiments.post_training.identity_t2 import capture as t2
from experiments.post_training.identity_t2.scorer_snapshot import NEGATION

POSITIVE_NAME = re.compile(r"\b(?:I am|I'm|my (?:model )?name is|I am called)\s+(?:\*\*|__)?Marin's Latest MoE\b", re.I)
POSITIVE_CREATOR = re.compile(
    r"\b(?:I(?:'m| am| was| have been)|Marin's Latest MoE)\b[^.!?\n]{0,180}?"
    r"\b(?:developed|trained|created|built|made)(?:\s+and\s+(?:developed|trained|created|built|made))?"
    r"\s+by\s+(?:the\s+)?Marin Community\b|"
    r"\b(?:the\s+)?Marin Community\s+(?:developed|trained|created|built|made)"
    r"(?:\s+and\s+(?:developed|trained|created|built|made))?\s+me\b",
    re.I,
)
FOREIGN_ROLE = re.compile(
    r"\b(?:developed|trained|created|built|made)(?:\s+and\s+(?:developed|trained|created|built|made))?"
    r"\s+by\s+(?:the\s+)?(?:Open Athena|Stanford|NVIDIA|AI2)\b",
    re.I,
)
DELIMITERS = re.compile(
    r"<\|[^\n]*?\|>|</?think>|</?tool_call>|</?s>|\[/?(?:INST|MASK|gMASK|sMASK|sop|eop)\]|<sop>|<eop>", re.I
)


def text_flags(reasoning: str, final: str) -> list[str]:
    """Conservatively reject exposed concrete faults; do not claim entailment proof."""
    flags = []
    normalized = final.replace("\u2019", "'")
    if not any(not NEGATION.search(match.group()) for match in POSITIVE_NAME.finditer(normalized)):
        flags.append("final_missing_positive_Held_name")
    if not any(not NEGATION.search(match.group()) for match in POSITIVE_CREATOR.finditer(normalized)):
        flags.append("final_missing_positive_Marin_Community_creator")
    for reason in identity_qc(reasoning, final, ""):
        if "noncanonical_" in reason:
            flags.append("reasoning_final_profile_or_creator_role_mismatch")
    for label, text in (("reasoning", reasoning), ("final", final)):
        if foreign_claim(text) or FOREIGN_ROLE.search(text):
            flags.append(label + ":foreign_self_or_creator_claim")
        if omitted_context(text) or REFERENCE.search(text):
            flags.append(label + ":omitted_system_instruction_reference")
        if DELIMITERS.search(text):
            flags.append(label + ":literal_control_delimiter")
        if re.search(r"\b\d", text):
            flags.append(label + ":unsupported_numeric_or_version_fact")
        for predicate in (
            LINEAGE,
            LINEAGE_V2,
            UNSUPPORTED_EXCLUSION,
            TRAINING_CONTRIBUTOR,
            ORGANIZATION_V2,
            ROLE_EXCLUSION_V2,
            UI_CAPABILITY_V2,
        ):
            if predicate.search(text):
                flags.append(label + ":unsupported_fact_or_role_prescreen")
        for clause in re.split(r"[.!?;]\s+|\b(?:but|however|whereas|yet)\b", text, flags=re.I):
            if not UNKNOWN.search(clause) and any(
                predicate.search(clause) for predicate in (ARCHITECTURE, LICENSE, DEPLOYMENT)
            ):
                flags.append(label + ":unsupported_architecture_license_deployment")
    return sorted(set(flags))


def checked_student(record: dict, tokenizer) -> tuple[dict | None, list[str]]:
    """Keep every accepted raw target ID in a system-free native student row."""
    candidate = record["candidate"]
    if record["request"] != t2.request_body(candidate, SOURCE_MODEL):
        raise ValueError("Actual A5 request body differs from frozen sampling/system/context")
    if record.get("fatal"):
        return None, ["capture_transport_or_protocol_failure"]
    if record.get("length_stop"):
        return None, ["model_length_stop"]
    ids = record["raw_token_ids"]
    if ids != t2.generated_ids(record["response"]):
        raise ValueError("Raw IDs differ from provider payload")
    if any(
        record[key] != t2.prompt_ids(tokenizer, candidate, teacher=True)
        for key in ("cpu_teacher_prompt_token_ids", "server_teacher_prompt_token_ids")
    ):
        raise ValueError("Teacher CPU/server/native prefix mismatch")
    if record["student_prompt_token_ids"] != t2.prompt_ids(tokenizer, candidate):
        raise ValueError("Student native prefix mismatch")
    choice = record["response"]["choices"][0]
    if (
        choice.get("finish_reason") != "stop"
        or ids[-1] != EOT
        or ids[0] != START_THINK
        or ids.count(START_THINK) != 1
        or ids.count(END_THINK) != 1
    ):
        return None, ["unclean_native_reasoning_final_protocol"]
    end = ids.index(END_THINK)
    reason_ids, final_ids = ids[1:end], ids[end + 1 : -1]
    if not reason_ids or not final_ids or set(reason_ids + final_ids) & SPECIAL_TOKEN_IDS:
        return None, ["empty_reasoning_final_or_nested_control"]
    fidelity = t2.roundtrip(tokenizer, candidate, ids)
    if not fidelity["ok"]:
        return None, ["raw_native_student_graft_failure"]
    decode = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
    reasoning, final = tokenizer.decode(reason_ids, **decode), tokenizer.decode(final_ids, **decode)
    flags = text_flags(reasoning, final)
    if flags:
        return None, flags
    prefix = record["student_prompt_token_ids"]
    if len(prefix) + len(ids) > 8192:
        return None, ["natural_target_exceeds_student8192_no_trim"]
    row = {
        "id": "pathb-a5-" + candidate["source_id"] + "-on",
        "source_id": candidate["source_id"],
        "family": candidate["family"],
        "enable_thinking": True,
        "messages": [*candidate["prompt"], {"role": "assistant", "content": fidelity["content"]}],
        "chat_template_kwargs": {"enable_thinking": True},
        "input_ids": [*prefix, *ids],
        "target_start": len(prefix),
        "assistant_masks": [0] * len(prefix) + [1] * len(ids),
        "supervised_tokens": len(ids),
        "weighted_supervised_tokens": len(ids),
        "final_text_tokens": sum(token not in LOW_WEIGHT_IDS for token in final_ids),
        "reasoning_text_tokens": len(reason_ids),
        "target_token_counts": dict(Counter(map(str, ids))),
        "semantic_qc_status": "PRESCREEN_PASS_PENDING_ROLE_AWARE_ADMISSION",
        "reasoning_policy": "coupled_v0.1_Held_profile_teacher_ON_replacement_new_final_no_system",
    }
    return row, []
