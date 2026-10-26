# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen scoring-only subset; no endpoint, retry or collection CLI.

Source SHA256: 6414e1be1712df6126f0f7da8317658f7ca2ab7fc35af8ad5a41c2950b259deb
Function bodies/constants retain the original AST; training overrides live outside.
"""

import re
from collections import Counter

from experiments.post_training.identity_t2 import scorer_snapshot as shared

ORIGINAL_SOURCE_SHA256 = "6414e1be1712df6126f0f7da8317658f7ca2ab7fc35af8ad5a41c2950b259deb"
ORIGINAL_NODE_AST_SHA256 = {
    "FILTER_VERSION": "2c956ade7e82c0b2ab6452363070ab444f0d37cbd597ee88ff1d4d04afdb143d",
    "FILTER_REVISION": "a9a39e1ee7c518ca1ab99dc7cebfa1e59659ee7587f2b08115cd2e086b165a57",
    "STUDENT_CONTEXT": "8ccee2efde2609cc0a8802d683a9c16eb81f8726517a31752f7993d51568d2c6",
    "LOOP_WINDOW_WORDS": "76a2dcd628704847d846753f625414d35cc4e50532a7453cfffd158484a2f23f",
    "LOOP_REPETITIONS": "6824427ca41f796122f1d3c1d79a47cb269cbc48d30dcac852a5e593a49fd5da",
    "TEACHER_SYSTEMS": "47e30f73dcebca51c6c64bca482fe3d3c58f961391e3c7d041afe4dee3994d49",
    "IDENTITY_CONTEXT_ATTRIBUTION": "93f55f6368fd5fa06332f2f8f761a3e5e5f5f4b516ee18f033c4b93e31304bd0",
    "POSITIVE_FIRST_PERSON_NAME": "610a948f91bcb9ece5b1f26b7a8d3045624ca96e54e21eb530053434cc04e734",
    "QUOTED_TEXT": "3d8e12d8803e1d0b377e5f5e5021c59cc218fccc7a35a7ed4a8c311478bfe3cf",
    "PLANNED_ANSWER": "6acd59d58b44bdc725f5fd67e24edd24f3c578e79b18ea9ee187cdc5b66155e0",
    "POSITIVE_FIRST_PERSON_CREATOR": "19e276dc4ff06270be5eee2b5b234c8939c32ce8e5b75e3f2479fd89ba581e55",
    "CREATOR_SURFACE": "6f01bc14f8301a77927ed10d5b8174e97261fdc7ebb4096e8e7cd8ef9a4cb055",
    "NAME_SURFACE": "9552a78f738ec2c8cbf507899972565681412bf33de470b1e8c0119f17ffa22d",
    "SELF_ALIAS": "25b45682e4d27ac0037f6b8d43ba5c2ede00d21bff404b922f07961f0032800f",
    "NEGATIVE_SELF_NAME": "d678e617c1b2951169952934634f5f63af95753480d2fe3cb24639ec1aa42d08",
    "TRAINING_OTHER_ENTITIES": "5ba2ee70a1ec93fb4b66de6cbd7e0ebe877d5fd49a849c047e93a6b989173911",
    "FOREIGN_SELF_IDENTITY": "650822c2f1ca47cca6cd65dcf8f14f46c01dd84652f2a0cfa2b7d4a4d353b67f",
    "UNSUPPORTED_IDENTITY_DETAIL": "ffe7b6769a5c619b04ef2667c62cc73a18b3552e85a64229935e63a51d2437f3",
    "NEUTRAL_NAMED_IDENTITY": "96a811b109d726a43549c804d0c30c51380e3252e13e9f0aff040a79898c6c0a",
    "NEUTRAL_SELF_OR_SOURCE": "f627b9be1a42f9647479c808fd8be7462218f0be94b17b413b070dd991a31e9f",
    "FIRST_PERSON_ANSWER": "1d3818e1fbf06196c709ce159dc8aabe5e87f9142910ed99928231cf935349c2",
    "intrinsic_reasoning": "4f196fea35677f55cf3c4c76ea5f877108cf0f2e667bfe14d7670221d039fea8",
    "identity_neutral_reasoning": "e341342b0f0c7865dc578327a0c8ae3e742a27e16a6b88396a32f2d76a7ee01e",
    "foreign_self_identity": "ce94b3c8bb1b579c55292cd59bcd36ff49617b56d72f9e5df49f4062acd557e1",
    "identity_decision": "9d5632939ad47f96f2e3e99f916783924c4bdc40cb2f7d7997cd8a9268dace9c",
}

FILTER_VERSION = 4


FILTER_REVISION = 2


STUDENT_CONTEXT = 8192


LOOP_WINDOW_WORDS = 8


LOOP_REPETITIONS = 8


TEACHER_SYSTEMS = {
    "canonical": shared.BASELINE_SYSTEM_PROMPT,
    "fact": "Identity: Marin Snowball, an AI model created by Open Athena.",
}


IDENTITY_CONTEXT_ATTRIBUTION = re.compile(
    r"\b(?:system|developer)(?:['\u2019]s)?\s+(?:instructions?|details)\b|"
    r"\b(?:system|developer)\b[^.\n]{0,120}"
    r"\b(?:prompt|message|says?|tells?|states?|identity|Marin|Snowball|Open[- ]?Athena)\b|"
    r"\b(?:prompt|instructions?)\b[^.\n]{0,120}\b(?:says?|tells?|states?|Marin|Snowball|Open[- ]?Athena)\b|"
    r"\b(?:as instructed|as requested|per the instructions|what I was told|given identity|provided identity)\b|"
    r"\b(?:I was|I have been|I've been|we were)\s+(?:told|instructed)\b|"
    r"\bIdentity\s*:|\b(?:the )?context\s*:|"
    r"\b(?:the user|they|you)\s+(?:says?|said|told|gave|defined|has defined|have defined|"
    r"has given|have given|describes?|is referencing)\b[^.!?\n]{0,180}"
    r"\b(?:Marin|Snowball|Open[- ]?Athena)\b|"
    r"\b(?:according to|based on)\s+(?:the )?(?:identity|description)\b|"
    r"\b(?:given|provided|defined|specified) identity\b|"
    r"\bidentity\s+(?:given|provided|says|statement|they have defined|you've been given)\b|"
    r"\bthe model they (?:have )?defined\b|"
    r"\b(?:the )?name (?:given|provided)\b|\bas per the description\b|"
    r"\b(?:we are to respond as|respond as Marin|they identify themselves as)\b|"
    r"\bthe user (?:gave|provided)\b|\b(?:these instructions|described as)\b|"
    r"\b(?:custom identity|custom model|pretend|role[- ]?play)\b|"
    r"\b(?:they|the user)\s+(?:have |has )?introduced themselves\b|"
    r"\b(?:the user|you)\s+(?:is|are)\s+Marin\s+Snowball\b|"
    r"\bthe user says\s+[\"']?I am\b|"
    r"\b(?:the user|they)\s+identif(?:y|ies)\s+(?:themselves\s+)?as\b|"
    r"\b(?:the first line|the initial (?:prompt|statement)|the user introduced me|"
    r"they mention being created|they refer to the assistant)\b|"
    r"\b(?:the )?user says\s*:\s*[\"']?I am\b",
    re.IGNORECASE,
)


POSITIVE_FIRST_PERSON_NAME = re.compile(
    r"\b(?:I am|I'm|we are|we're|my (?:model )?name is|our (?:model )?name is)\s+"
    r"(?:(?:actually|really)\s+)?(?:an? AI (?:model|assistant) (?:named|called)\s+)?Marin\s+Snowball\b",
    re.IGNORECASE,
)


QUOTED_TEXT = re.compile(r"\"[^\"\n]*\"|\u201c[^\u201d\n]*\u201d|(?<!\w)'[^'\n]*'(?!\w)")


PLANNED_ANSWER = re.compile(
    r"(?:^|\n)\s*(?:Thus(?:\s+(?:final\s+)?answer)?|(?:Final\s+)?Answer|I'll respond|Response)\s*:",
    re.IGNORECASE,
)


POSITIVE_FIRST_PERSON_CREATOR = re.compile(
    r"\b(?:I am|I'm|I was|we are|we're|we were)\s+"
    r"(?:(?:an?|the)\s+(?:AI |language )?(?:model|assistant)\s+)?"
    r"(?:created|made|built|developed|designed)\s+by\s+(?:\*\*|__)?Open Athena\b|"
    r"\b(?:my|our)\s+(?:creator|developer|maker)\s+is\s+(?:\*\*|__)?Open Athena\b|"
    r"\bOpen Athena\s+(?:created|made|built|developed|designed)\s+(?:me|us)\b",
    re.IGNORECASE,
)


CREATOR_SURFACE = re.compile(r"\bOpen[\s_\-\u200b\u2060]*Athena\b", re.IGNORECASE)


NAME_SURFACE = re.compile(r"\bMarin[\s_\-\u200b\u2060]*Snowball\b", re.IGNORECASE)


SELF_ALIAS = re.compile(
    r"\b(?:Athena AI|another alias|other alias|call me whatever|call me anything)\b|"
    r"\b(?:call me|my name is|I'm called|I am called)\s+[\"']?(?:\*\*|__)?"
    r"(?:Athena|Open Athena|Snowplow|Snowflake)\b|"
    r"\b(?:like|maybe|perhaps)\s+[\"']?Athena\b",
    re.IGNORECASE,
)


NEGATIVE_SELF_NAME = re.compile(
    r"\b(?:I|we)\s+(?:don'?t|do not|have no|lack)[^.\n]{0,70}\b(?:name|identity)\b", re.IGNORECASE
)


TRAINING_OTHER_ENTITIES = shared.OTHER_ENTITIES + r"|Copilot|Microsoft|Bard|Alexa|Amazon|Siri|Apple|Grok|xAI|Perplexity"


FOREIGN_SELF_IDENTITY = re.compile(
    r"\b(?:As|I am|I'm|I was|we are|we're|we were|my creator is|my developer is|call me|"
    r"this model is|the assistant is)\b"
    rf"[^.\n]{{0,60}}?\b({TRAINING_OTHER_ENTITIES})\b",
    re.IGNORECASE,
)


UNSUPPORTED_IDENTITY_DETAIL = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:based on|built on|underlying architecture|parent company|"
    r"GPT architecture|GPT-?\d architecture|billion parameters?|release date|nonprofit|non-profit|"
    r"model parameters?|training (?:dates?|methodology)|unique features|model version)\b",
    re.IGNORECASE,
)


NEUTRAL_NAMED_IDENTITY = re.compile(
    rf"\b(?:Marin|Snowball|Open\s*[-]?\s*Athena|{TRAINING_OTHER_ENTITIES})\b", re.IGNORECASE
)


NEUTRAL_SELF_OR_SOURCE = re.compile(
    r"\b(?:I am|I'm|we are|we're|I was|we were|my name|our name|my creator|our creator|"
    r"(?:I|we)\s+(?:don'?t|do not|have no|lack|possess no)[^.\n]{0,60}\b(?:name|identity)|"
    r"as an? AI|the assistant is|this model is|the model is|"
    r"the (?:model|assistant)'s name|system|developer|instructions?|prompt|context|description|"
    r"given|provided|roles?|personas?|respond as|act as|pretend|custom|fictional|hypothetical)\b",
    re.IGNORECASE,
)


FIRST_PERSON_ANSWER = re.compile(
    r"\b(?:I am|I'm|we are|we're|my (?:model )?name is|our (?:model )?name is|(?:you can )?call me)\s+"
    r"(?:\*\*|__)?Marin\s+Snowball\b|"
    r"\b(?:I am|I'm|I was|we are|we're|we were)\s+"
    r"(?:an? AI (?:model|assistant)\s+)?(?:created|made|built|developed|designed)\s+by\s+"
    r"(?:\*\*|__)?Open Athena\b|" + POSITIVE_FIRST_PERSON_CREATOR.pattern,
    re.IGNORECASE,
)


def intrinsic_reasoning(text: str) -> str:
    """Exclude proposed replies; quoted entity labels alone are not quoted self-statements."""
    normalized = text.replace("\u2019", "'")
    prefix = PLANNED_ANSWER.split(normalized, maxsplit=1)[0]

    def mask_quote(match: re.Match) -> str:
        label = match.group()[1:-1]
        return label if re.fullmatch(r"(?:Marin Snowball|Open Athena)", label) else ""

    return QUOTED_TEXT.sub(mask_quote, prefix)


def identity_neutral_reasoning(text: str) -> bool:
    normalized = text.replace("\u2019", "'")
    return bool(normalized.strip()) and not (
        NEUTRAL_NAMED_IDENTITY.search(normalized) or NEUTRAL_SELF_OR_SOURCE.search(normalized)
    )


def foreign_self_identity(text: str) -> bool:
    normalized = text.replace("\u2019", "'")
    return any(
        not shared.NEGATION.search(normalized[match.start() : match.start(1)])
        for match in FOREIGN_SELF_IDENTITY.finditer(normalized)
    )


def identity_decision(
    seed: dict, answer: str, reasoning: str, finish_reason: str, sequence_tokens: int, *, thinking: bool = True
) -> dict:
    """Check actual answer AND reasoning; forbid identity attributed to omitted teacher instructions."""
    row = shared.Row(
        id=seed["id"],
        set="identity",
        category="training",
        group="ctxdistill",
        split="train",
        messages=seed["messages"],
        expect=seed["expect"],
    )
    answer_labels = shared.prescreen_labels(row, answer)
    reasoning_labels = shared.prescreen_labels(row, reasoning)
    self_reasoning = intrinsic_reasoning(reasoning)
    creator_family = seed.get("training_family") == "creator" or seed["expect"] == ["creator"]
    positive_name = bool(POSITIVE_FIRST_PERSON_NAME.search(self_reasoning))
    positive_creator = creator_family and bool(POSITIVE_FIRST_PERSON_CREATOR.search(self_reasoning))
    positive = positive_name or positive_creator
    # No pure-neutral reasoning for negation: the false identity premise must be resolved.
    negation = seed.get("type") == "negation" or set(seed["expect"]) == {"name", "creator"}
    neutral = not negation and identity_neutral_reasoning(reasoning)
    reasoning_class = ("positive" if positive else "neutral" if neutral else None) if thinking else "thinking_off"
    reasons = []
    if finish_reason != "stop":
        reasons.append("incomplete_finish")
    if not answer.strip():
        reasons.append("empty_answer")
    if thinking and not reasoning.strip():
        reasons.append("empty_reasoning")
    if not thinking and reasoning.strip():
        reasons.append("unexpected_reasoning_in_thinking_off")
    if not answer_labels["prescreen_pass"]:
        reasons.append("wrong_answer_identity")
    if thinking and reasoning_class != "neutral" and not reasoning_labels["prescreen_pass"]:
        reasons.append("wrong_reasoning_identity")
    for span in (reasoning, answer):
        words = span.split()
        windows = Counter(
            tuple(words[index : index + LOOP_WINDOW_WORDS]) for index in range(len(words) - LOOP_WINDOW_WORDS + 1)
        )
        if max(windows.values(), default=0) >= LOOP_REPETITIONS:
            reasons.append("repetitive_trace")
            break
    if sequence_tokens > STUDENT_CONTEXT:
        reasons.append("student_sequence_too_long")
    before_attribution_filter = not reasons
    spans = (reasoning, answer)
    attribution = any(
        IDENTITY_CONTEXT_ATTRIBUTION.search(span)
        or any(system.casefold() in " ".join(span.split()).casefold() for system in TEACHER_SYSTEMS.values())
        for span in spans
    )
    if attribution:
        reasons.append("identity_attributed_to_omitted_context")
    if reasoning_class is None:
        reasons.append("no_positive_first_person_name_in_reasoning")
    if reasoning_class in {"neutral", "thinking_off"} and not FIRST_PERSON_ANSWER.search(answer.replace("\u2019", "'")):
        reasons.append("first_person_correct_answer_required")
    if any(match.group() != "Open Athena" for span in spans for match in CREATOR_SURFACE.finditer(span)):
        reasons.append("noncanonical_creator_spelling")
    if any(match.group() != "Marin Snowball" for span in spans for match in NAME_SURFACE.finditer(span)):
        reasons.append("noncanonical_name_spelling")
    if any(SELF_ALIAS.search(span) for span in spans):
        reasons.append("invented_self_alias")
    if any(NEGATIVE_SELF_NAME.search(span.replace("\u2019", "'")) for span in spans):
        reasons.append("negative_self_identity_predicate")
    if any(foreign_self_identity(span) for span in spans):
        reasons.append("foreign_self_identity_training_filter")
    if any(UNSUPPORTED_IDENTITY_DETAIL.search(span) for span in spans):
        reasons.append("unsupported_secondary_identity_fact")
    return {
        "answer_labels": answer_labels,
        "reasoning_labels": reasoning_labels,
        "passes_before_attribution_filter": before_attribution_filter,
        "context_attribution": attribution,
        "reasoning_class": reasoning_class,
        "rejection_reasons": reasons,
    }
