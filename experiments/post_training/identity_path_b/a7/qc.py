# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A7 raw native-span admission, with unchanged A5b semantic prescreens."""

import re
from collections import Counter

from experiments.post_training.identity_path_b.a5.qc import POSITIVE_NAME
from experiments.post_training.identity_path_b.a5.qc import text_flags as a5_text_flags
from experiments.post_training.identity_path_b.a7.data import (
    END_THINK,
    START_THINK,
    request_body,
    student_graft,
    teacher_prompt_ids,
)
from experiments.post_training.identity_path_b.data import LOW_WEIGHT_IDS, SPECIAL_TOKEN_IDS
from experiments.post_training.identity_path_b.reasoning import identity_qc
from experiments.post_training.identity_t2 import capture as t2
from experiments.post_training.identity_t2.scorer_snapshot import NEGATION

REASONING_REFERENCE = re.compile(r"\b(?:systems?|polic(?:y|ies)|instructions?|prompts?)\b", re.I)


CANONICAL_NAME = re.escape("Marin's Latest MoE")
NAME_DISCLAIMER = re.compile(
    CANONICAL_NAME + r"\s+(?:is|was)\s+(?:not\s+(?:my|our|the)\s+(?:model\s+)?name|"
    r"(?:an?\s+)?(?:another|different)\s+model|"
    r"(?:an?\s+)?(?:ChatGPT|Gemini|Claude|OpenAI|Qwen|DeepSeek|Marin Snowball)\b)",
    re.I,
)
LEADING_NAME = re.compile(r"^\s*" + CANONICAL_NAME + r"\b(?=\s*(?:[,.:;—\u2013-]|$|(?:is|was|has been|here)\b))", re.I)
HYPOTHETICAL = re.compile(r"\b(?:if|suppose|imagine|hypothetical(?:ly)?|would|could|might|quoted?|example)\b", re.I)

DENIED_NAME_CLAUSE = re.compile(
    r"\b(?:den(?:y|ies|ied|ying)|false|untrue|lie|incorrect|reject(?:ed|ing)?|refut(?:e|ed|ing)|cannot|whether)\b", re.I
)

FOREIGN_ENTITY = (
    r"(?:ChatGPT|GPT[- ]?4|GPT[- ]?5|Gemini|Claude|Llama|Qwen|DeepSeek|Kimi|GLM|OpenAI|Google|"
    r"Anthropic|Meta|Alibaba|Moonshot|ZAI|Open Athena|Stanford|NVIDIA|AI2)"
)
SEPARATE_FOREIGN_DENIAL = re.compile(
    r"^\s*(?:No\s*[,—]\s*)?(?:I(?:'m| am)\s+(?:not|neither)\s+(?:an?\s+)?"
    + FOREIGN_ENTITY
    + r"|I\s+(?:was|am|have been)\s+(?:not|never)\s+(?:developed|trained|created|built|made)"
    r"(?:\s+and\s+(?:developed|trained|created|built|made))?\s+by\s+(?:the\s+)?"
    + FOREIGN_ENTITY
    + r")\s*(?:,\s*(?:(?:but|and|however|rather|instead)\s+)?|\s+(?:but|and|however|rather|instead)\s+)",
    re.I,
)


def affirmative_clause_prefix(before: str) -> str:
    """Exclude separately completed denials of foreign identity from this name clause."""
    while match := SEPARATE_FOREIGN_DENIAL.match(before):
        before = before[match.end() :]
    return before


def positive_name(final: str) -> bool:
    """Affirmative canonical identity wording; formatting is normalized for this predicate only."""
    text = final.replace("\u2019", "'").replace("**", "").replace("__", "")
    if NAME_DISCLAIMER.search(text):
        return False
    for match in POSITIVE_NAME.finditer(text):
        before = re.split(r"[.!?;\n]", text[: match.start()])[-1]
        before = affirmative_clause_prefix(before)
        clause = before + re.split(r"[.!?\n]", text[match.start() :])[0]
        if (
            not HYPOTHETICAL.search(clause)
            and not NEGATION.search(before)
            and not DENIED_NAME_CLAUSE.search(before)
            and not before.lstrip().startswith(">")
            and not before.rstrip().endswith(("'", '"', "\u201c", "\u2018", "`"))
        ):
            return True
    leading = LEADING_NAME.match(text)
    if leading:
        clause = re.split(r"[.!?\n]", text)[0]
        return not HYPOTHETICAL.search(clause) and not re.search(r"\bnot\s+(?:me|my\s+name)\b", clause, re.I)
    return False


def text_flags(reasoning: str, final: str) -> list[str]:
    profile = "reasoning_final_profile_or_creator_role_mismatch"
    flags = [
        flag for flag in a5_text_flags(reasoning, final) if flag not in ("final_missing_positive_Held_name", profile)
    ]
    normalized = [text.replace("\u2019", "'").replace("**", "").replace("__", "") for text in (reasoning, final)]
    if any("noncanonical_" in reason for reason in identity_qc(normalized[0], normalized[1], "")):
        flags.append(profile)
    if not positive_name(final):
        flags.append("final_missing_positive_Held_name")
    if REASONING_REFERENCE.search(reasoning):
        flags.append("reasoning:literal_system_policy_instruction_prompt")
    return sorted(set(flags))


def checked_student(record: dict, tokenizer) -> tuple[dict | None, list[str]]:
    candidate = record["candidate"]
    if record["request"] != request_body(candidate):
        raise ValueError("Actual A7 request differs from exact teacher/context/sampling/prefill")
    if record.get("fatal"):
        return None, ["capture_transport_or_protocol_failure"]
    ids = record["raw_token_ids"]
    if ids != t2.generated_ids(record["response"]):
        raise ValueError("A7 generated IDs differ from original provider payload")
    teacher = teacher_prompt_ids(tokenizer, candidate)
    if any(record[key] != teacher for key in ("cpu_teacher_prompt_token_ids", "server_teacher_prompt_token_ids")):
        raise ValueError("Native teacher/custom-template CPU/server prefix mismatch")
    if record["response"].get("prompt_token_ids") != teacher:
        raise ValueError("Provider completion prompt IDs differ from native prefixed teacher IDs")
    prefix = t2.prompt_ids(tokenizer, candidate)
    if record["student_prompt_token_ids"] != prefix:
        raise ValueError("Native student base prefix changed")
    if record.get("length_stop"):
        return None, ["model_length_stop"]
    choice = record["response"]["choices"][0]
    if (
        choice.get("finish_reason") != "stop"
        or not ids
        or ids[-1] != 128009
        or START_THINK in ids
        or ids.count(END_THINK) != 1
    ):
        return None, ["unclean_or_unclosed_prefilled_native_reasoning_final_protocol"]
    end = ids.index(END_THINK)
    reason_ids, final_ids = ids[:end], ids[end + 1 : -1]
    if not reason_ids or not final_ids or set(reason_ids + final_ids) & SPECIAL_TOKEN_IDS:
        return None, ["empty_reasoning_final_or_nested_control"]
    graft = student_graft(tokenizer, candidate, ids)
    if record["prefill_student_graft"] != graft:
        raise ValueError("Stored A7 native prefill/student graft proof differs")
    if not graft["ok"]:
        return None, ["raw_native_prefill_student_graft_failure"]
    decode = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
    reasoning, final = tokenizer.decode(reason_ids, **decode), tokenizer.decode(final_ids, **decode)
    flags = text_flags(reasoning, final)
    if flags:
        return None, sorted(set(flags))
    target = graft["student_target_token_ids"]
    if len(prefix) + len(target) > 8192:
        return None, ["natural_target_exceeds_student8192_no_trim"]
    return (
        dict(
            id="pathb-a7-" + candidate["source_id"] + "-on",
            source_id=candidate["source_id"],
            family=candidate["family"],
            enable_thinking=True,
            messages=[*candidate["prompt"], {"role": "assistant", "content": graft["content"]}],
            chat_template_kwargs={"enable_thinking": True},
            input_ids=prefix + target,
            target_start=len(prefix),
            assistant_masks=[0] * len(prefix) + [1] * len(target),
            supervised_tokens=len(target),
            weighted_supervised_tokens=len(target),
            final_text_tokens=sum(token not in LOW_WEIGHT_IDS for token in final_ids),
            reasoning_text_tokens=len(reason_ids),
            target_token_counts=dict(Counter(map(str, target))),
            semantic_qc_status="PRESCREEN_PASS_PENDING_ROLE_AWARE_READ",
            reasoning_policy="A3_no_system_forced_native_think_teacher_pilot_only",
        ),
        [],
    )
