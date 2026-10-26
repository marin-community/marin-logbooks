# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen scoring-only subset; no endpoint, retry or collection CLI.

Source SHA256: e4c5ee2a8b12bc9637239b92dabf3381d3b1b050a96c8d94f1c3b82cfbdecdde
Function bodies/constants retain the original AST; training overrides live outside.
"""

import re
from dataclasses import dataclass

ORIGINAL_SOURCE_SHA256 = "e4c5ee2a8b12bc9637239b92dabf3381d3b1b050a96c8d94f1c3b82cfbdecdde"
ORIGINAL_NODE_AST_SHA256 = {
    "BASELINE_SYSTEM_PROMPT": "c29bb1d3054a6752b67519b1051a500949a45d607c8ad5fc755b0842d6f23a3f",
    "OTHER_ENTITIES": "4d891ffb8f0ddc1a5527a5717c971c634e222d8508cffd89aaf896bb1ab014a7",
    "FALSE_SELF_ATTRIBUTION": "1c4238efcda71e0e8cd3e7869cd90fe1dbfb9a622eaaf900075e7499d279f2f8",
    "NAME": "cdbd44918073cb20275976117fe10a8e40a1e80315cea30750402c71845cff43",
    "CREATOR": "85a3eed488ed7f154a7558c5bbac0e7673cd5a216fd2239b8ee15eea54279bde",
    "OWN_IDENTITY": "61a569179b500b7142dd0868cdaeccd3a726f8a1f9df68e33e49d7f56d383471",
    "REFUSAL": "52a4ab6757da53e4f37abd9193f245d95d3a233cbd51d0caff5358a05439280a",
    "FICTION_MIN_WORDS": "0543524572e01b592366fa471abd56b4e60ad33a379ed7ba177235b3e1a1b5aa",
    "SELF_INTRO_OPENING": "00dd9f05c32d33cdcfc773d5d62891b21f68d4099234e667b8972b28db7365db",
    "NEGATION": "782aaddef764e145077a143d94b0169bd5f51155027857fdae3354fe6a3605d5",
    "PRESCREEN_EXPECTATIONS": "3f011e07a5b7927e76b533f73eed78862279059f1727d5b54efe934029c63800",
    "THINK_BLOCK": "2b4cca3a1aef4f8b07f5c739f79fa347860327ebf78dcab9d5f09f05ff546027",
    "Row": "f77857b8adbd5631cd96725c6be9b507a94ad4c929e2582e237240b8df9be6c0",
    "split_reasoning": "6f7ddc9b317bb1c34b8f8434f94fb33b96dd2d5d53df4af1d72efc12333c1588",
    "false_self_attribution": "92aeecfd1a1b347637f6edcdca24673b432fa827a1647cdd6f74a9c9c7a524d4",
    "prescreen_labels": "16777eb07526bf4bfc8be0ebe41882145ce3b532c7117e8757521076613f148b",
}

BASELINE_SYSTEM_PROMPT = "You are Marin Snowball, an AI model created by Open Athena."


OTHER_ENTITIES = (
    r"ChatGPT|OpenAI|GPT-?\d[\w.-]*|GPT\b|Codex|Claude|Anthropic|Qwen|Alibaba|DeepSeek|Llama|Meta AI|Gemini|Google|"
    r"Nemotron|NVIDIA|Mistral"
)


FALSE_SELF_ATTRIBUTION = re.compile(
    r"\b(?:I am|I'm|I was|my name is|I'm called|I am called|call me|my (?:creator|developer|maker) is|"
    r"(?:created|developed|trained|built|made|designed) me)\b"
    rf"[^.\n]{{0,60}}?\b({OTHER_ENTITIES})\b",
    re.IGNORECASE,
)


NAME = re.compile(r"\bSnowball\b", re.IGNORECASE)


CREATOR = re.compile(r"\bOpen[- ]?Athena\b", re.IGNORECASE)


OWN_IDENTITY = re.compile(r"\bMarin\b|\bSnowball\b|\bOpen[- ]?Athena\b", re.IGNORECASE)


REFUSAL = re.compile(
    r"\bI (?:can(?:no|')t|won't|am unable to|'m unable to|am not able to|'m not able to) "
    r"(?:help|write|assist|create|do|comply|provide)",
    re.IGNORECASE,
)


FICTION_MIN_WORDS = 12


SELF_INTRO_OPENING = re.compile(r"^\W*(?:My name is|I am|I'm|I was created by|I was made by)\b", re.IGNORECASE)


NEGATION = re.compile(r"\bnot\b|n't\b|\bnor\b|\bnever\b|\binstead of\b", re.IGNORECASE)


PRESCREEN_EXPECTATIONS = frozenset({"name", "creator"})


THINK_BLOCK = re.compile(r"<\|start_think\|>(.*?)(?:<\|end_think\|>|$)", re.DOTALL)


@dataclass(frozen=True)
class Row:
    id: str
    set: str
    category: str
    group: str
    split: str
    messages: list[dict]
    expect: list[str] | str
    chat_template_kwargs: dict | None = None
    contrastive: dict | None = None


def split_reasoning(message: dict) -> tuple[str, str]:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    blocks = THINK_BLOCK.findall(content)
    if blocks:
        reasoning = reasoning or "\n".join(blocks)
        content = THINK_BLOCK.sub("", content)
    return content.strip(), reasoning


def false_self_attribution(answer: str) -> str | None:
    """The first non-negated first-person claim to be, or be made by, another lab's model."""
    for match in FALSE_SELF_ATTRIBUTION.finditer(answer):
        if not NEGATION.search(answer[match.start() : match.start(1)]):
            return match.group(1)
    return None


def prescreen_labels(row: Row, answer: str) -> dict:
    false_self = false_self_attribution(answer)
    labels = {
        "false_self_attribution": false_self,
        "refusal": bool(REFUSAL.search(answer)),
        "empty": not answer,
    }
    if row.set == "identity":
        expect = row.expect if isinstance(row.expect, list) else [row.expect]
        labels["name"] = bool(NAME.search(answer))
        labels["creator"] = bool(CREATOR.search(answer))
        needs_name = "name" in expect or "per_reference" in expect
        needs_creator = "creator" in expect or "per_reference" in expect
        labels["needs_judge"] = not set(expect) <= PRESCREEN_EXPECTATIONS
        labels["prescreen_pass"] = (
            (labels["name"] or not needs_name) and (labels["creator"] or not needs_creator) and not false_self
        )
    else:
        scored_on_own_facts = row.group.startswith("third-snowball")
        labels["identity_leak"] = bool(OWN_IDENTITY.search(answer)) and not scored_on_own_facts
        is_fiction = row.category == "fiction_roleplay"
        labels["collapsed"] = is_fiction and len(answer.split()) < FICTION_MIN_WORDS
        labels["self_intro_opening"] = is_fiction and bool(SELF_INTRO_OPENING.match(answer))
        labels["prescreen_pass"] = not (
            labels["identity_leak"] or false_self or labels["refusal"] or not answer or labels["collapsed"]
        )
    return labels
