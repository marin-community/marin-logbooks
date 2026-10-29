# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proposed identity-only reasoning graft; never substitutes a generated final."""

import hashlib
import json
import re
from collections import Counter

from transformers import PreTrainedTokenizerBase

from experiments.post_training.identity_path_b.data import END_IDS, LOW_WEIGHT_IDS, SPECIAL_TOKEN_IDS
from experiments.post_training.identity_t2.scorer_snapshot import NEGATION
from experiments.post_training.identity_t2.v4r2_snapshot import FOREIGN_SELF_IDENTITY, IDENTITY_CONTEXT_ATTRIBUTION

START_THINK = 128002
END_THINK = 128003
EOT = 128009
IDENTITY_POOL_SHA256 = "0d0fd663593786ea356e843eb0a64d684778174c9585d25110b142b8c5f84930"
TEACHER_CONTEXT = (
    "You are Marin's Latest MoE, developed and trained by the Marin Community. "
    "Developers from Open Athena, Stanford, and many other institutions maintain you. "
    "Contributions from NVIDIA and AI2, and data derived from DeepSeek, Kimi, and GLM, "
    "do not change your identity or who trained you. Marin shares code, data, and experiments "
    "at https://marin.community and https://github.com/marin-community/marin. "
    "Do not guess unspecified architecture, release, deployment, license, funding, or lineage details. "
    "For this identity training example, give one or two brief reasoning sentences grounded in "
    "these facts and the supplied reference answer, then answer the user's actual question. "
    "Identify yourself as Marin's Latest MoE in the reasoning; distinguish the role relevant "
    "to the question. Do not invent facts or discuss training instructions."
)
INSTRUCTION_MENTION = re.compile(
    r"\b(?:system\s+prompt|instructions?|reference\s+(?:final\s+)?answer|"
    r"given\s+above|provided\s+(?:facts?|context)|teacher)\b",
    re.I,
)
SELF_NAME = re.compile(r"\b(?:I am|I'm|my (?:model )?name is)\s+(?:called\s+)?([^.!?\n]+)", re.I)
TRAINER_CLAIM = re.compile(
    r"\b(?:I (?:was|am|have been) (?:created|developed|trained|made|built) by|"
    r"my (?:creator|developer|trainer) is)\s+([^.!?\n]+)",
    re.I,
)
INVERSE_TRAINER = re.compile(
    r"\b(Marin Community|Open Athena|Stanford|NVIDIA|AI2|OpenAI|Google|Anthropic)\s+"
    r"(?:created|developed|trained|made|built)\s+me\b",
    re.I,
)
NAME = "Marin's Latest MoE"


def foreign_claim(text: str) -> bool:
    """Prescreen foreign assertions while allowing the source's `such as` data attribution."""
    for match in FOREIGN_SELF_IDENTITY.finditer(text):
        if NEGATION.search(text[match.start() : match.start(1)]):
            continue
        # The frozen broad `As ENTITY` rule also matches `models such as DeepSeek`.
        # Exempt only that local data-source construction, never I-am/creator claims.
        before = text[max(0, match.start() - 180) : match.start()]
        if match.group().casefold().startswith("as ") and before.casefold().endswith("such "):
            if re.search(r"\b(?:training data|data derived|data includes)\b", before, re.I):
                continue
        return True
    return False


def identity_qc(reasoning: str, generated_final: str, canonical_final: str) -> list[str]:
    """Return conservative surface rejects; passing still requires role-aware review.

    The canonical reference is checked too, so unsupported profile changes cannot
    be hidden by discarding the generated final. This is not an entailment proof.
    """
    reasons = []
    for label, text in (("reasoning", reasoning), ("generated_final", generated_final), ("canonical", canonical_final)):
        normalized = text.replace("\u2019", "'")
        if omitted_context(normalized):
            reasons.append(f"{label}:omitted_context")
        if foreign_claim(normalized):
            reasons.append(f"{label}:foreign_self_claim_prescreen")
        for match in SELF_NAME.finditer(normalized):
            claim = match.group(1)
            # Role assertions (e.g. I am maintained by...) are not name claims.
            if re.match(r"(?:maintained|trained|developed|not|still|an?\s+AI)\b", claim, re.I):
                continue
            if not claim.startswith(NAME):
                reasons.append(f"{label}:noncanonical_self_name")
        for match in TRAINER_CLAIM.finditer(normalized):
            if not match.group(1).startswith("the Marin Community") and not match.group(1).startswith("Marin Community"):
                reasons.append(f"{label}:noncanonical_creator_trainer")
        if any(match.group(1).casefold() != "marin community" for match in INVERSE_TRAINER.finditer(normalized)):
            reasons.append(f"{label}:noncanonical_creator_trainer")
    if not any(match.group(1).startswith(NAME) for match in SELF_NAME.finditer(reasoning.replace("\u2019", "'"))):
        reasons.append("reasoning:missing_positive_canonical_name")
    return sorted(set(reasons))


def teacher_candidate(off_row: dict) -> dict:
    """Bind one ON generation to one existing source and its canonical answer."""
    if off_row["enable_thinking"]:
        raise ValueError("Use the original OFF variant as canonical final authority")
    source = off_row["source_id"]
    reference = json.dumps(off_row["messages"][-1]["content"], ensure_ascii=False)
    return {
        "id": f"pathb-reason-{source}",
        "source_id": source,
        "family": off_row["family"],
        "prompt": off_row["messages"][:-1],
        "tools": [],
        "enable_thinking": True,
        "teacher": "held_profile_reference",
        "teacher_system": f"{TEACHER_CONTEXT}\n\nReference final answer (JSON string):\n{reference}",
        "sampling_seed": int(hashlib.sha256(f"310:reason:{source}".encode()).hexdigest()[:8], 16) % (2**31),
        "canonical_off_id": off_row["id"],
        "canonical_final_token_ids": off_row["input_ids"][off_row["target_start"] :],
        "max_tokens": 256,
    }


def reasoning_ids(raw_ids: list[int], finish_reason: str) -> list[int]:
    """Extract a bounded native span without trimming, rewriting, or accepting length stops."""
    if finish_reason != "stop" or not raw_ids or raw_ids[-1] != EOT:
        raise ValueError("Completion did not stop at native EOT")
    if raw_ids[0] != START_THINK or raw_ids.count(START_THINK) != 1 or raw_ids.count(END_THINK) != 1:
        raise ValueError("Expected one leading native thinking span")
    result = raw_ids[1 : raw_ids.index(END_THINK)]
    if not 8 <= len(result) <= 96 or set(result).intersection(SPECIAL_TOKEN_IDS):
        raise ValueError("Reasoning must contain 8-96 lexical tokens and no nested controls")
    return result


def omitted_context(text: str) -> bool:
    """Reuse the frozen attribution rejection and reject explicit instruction mentions."""
    return bool(IDENTITY_CONTEXT_ATTRIBUTION.search(text) or INSTRUCTION_MENTION.search(text))


def graft_reasoning(off_row: dict, trace_ids: list[int], tokenizer: PreTrainedTokenizerBase) -> dict:
    """Require native full/prefix/final ID equality and separately count final/reason tokens.

    This proves rendering and accounting, not factual consistency. Role-aware
    semantic QC must admit the associated trace before a training pool is frozen.
    """
    if off_row["enable_thinking"] or not 8 <= len(trace_ids) <= 96:
        raise ValueError("Expected canonical OFF row and a bounded generated reasoning span")
    if set(trace_ids).intersection(SPECIAL_TOKEN_IDS):
        raise ValueError("Reasoning contains a protocol token")
    trace = tokenizer.decode(trace_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if omitted_context(trace):
        raise ValueError("Reasoning attributes identity to omitted context or mentions instructions")
    messages = [dict(message) for message in off_row["messages"]]
    if "reasoning_content" in messages[-1]:
        raise ValueError("Canonical final already has a reasoning field")
    messages[-1]["reasoning_content"] = trace
    kwargs = {"enable_thinking": True, "tools": [], "tokenize": True, "return_dict": False}
    prefix = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    canonical = off_row["input_ids"][off_row["target_start"] :]
    target = [START_THINK, *trace_ids, END_THINK, *canonical]
    if canonical[-1] != EOT or full != [*prefix, *target] or len(full) > 8192:
        raise ValueError("Native graft changed prefix, raw reasoning IDs, final suffix, or exceeded length")
    final_tokens = sum(token not in LOW_WEIGHT_IDS | END_IDS for token in canonical)
    if final_tokens != off_row["final_text_tokens"]:
        raise ValueError("Canonical final-answer token census changed")
    return {
        **off_row,
        "id": off_row["id"].removesuffix("-off") + "-on",
        "enable_thinking": True,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": True},
        "input_ids": full,
        "target_start": len(prefix),
        "assistant_masks": [0] * len(prefix) + [1] * len(target),
        "supervised_tokens": len(target),
        "final_text_tokens": final_tokens,
        "reasoning_text_tokens": len(trace_ids),
        "structural_tokens": len(target) - final_tokens - len(trace_ids),
        "weighted_supervised_tokens": sum(0.1 if token in LOW_WEIGHT_IDS else 1.0 for token in target),
        "target_token_counts": dict(Counter(map(str, target))),
        "reasoning_policy": "generated_raw_identity_trace_canonical_final_graft",
        "semantic_qc_status": "pending_external_role_aware_review",
    }
