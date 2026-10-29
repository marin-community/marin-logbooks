# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Only the fixed52 identity/14 third-person OFF checkpoint diagnostics.

Row/rendering/label primitives below are copied from identity_eval.py at
9e3857f8. Endpoint transport is explicitly ONE attempt, unlike its four-attempt
historical transport. Historical regex fields remain prescreens, not grades.
"""

import hashlib
import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import requests

IDENTITY_SHA256 = "4569cb3a6e94330b624d7d0f1538ccd54c00085c5d537bdf1d583b5008dc64ee"
CONTROLS_SHA256 = "3b1f3328608bf21fe58680485030f67d70492e29bf350ec8d74e81ba9a611ad6"
CATEGORIES = frozenset(
    {"delayed", "direct_creator", "direct_name", "false_premise", "generic_system", "indirect", "multilingual"}
)
SEMANTIC_RULES_SHA256 = "1fb88f637b01ded727d39bf5a135d24ffb233e9c8d4a3561c628333e95beb605"

BASELINE_SYSTEM_PROMPT = "You are Marin Snowball, an AI model created by Open Athena."

FAIRY_PLACEHOLDER = "(model's own story goes here)"

REQUEST_TIMEOUT = 900

IDENTITY_MAX_TOKENS = 1024

CONTROL_MAX_TOKENS = 2048

THINKING_MAX_TOKENS = 4096

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


class Condition(StrEnum):
    NO_SYSTEM = "no_system"
    SYSTEM_BASELINE = "system_baseline"


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


def load_rows(prompt_dir: Path, split: str) -> list[Row]:
    rows = []
    for name in ("identity.jsonl", "controls.jsonl"):
        for line in (prompt_dir / name).read_text().splitlines():
            raw = json.loads(line)
            row = Row(
                id=raw["id"],
                set=raw["set"],
                category=raw["category"],
                group=raw["group"],
                split=raw["split"],
                messages=raw["messages"],
                expect=raw["expect"],
                chat_template_kwargs=raw.get("chat_template_kwargs"),
                contrastive=raw.get("contrastive"),
            )
            if split == "all" or row.split == split:
                rows.append(row)
    return rows


def with_condition(messages: list[dict], condition: Condition) -> list[dict] | None:
    """Return the messages for a condition, or None when the row is not run under it."""
    if condition is Condition.NO_SYSTEM:
        return messages
    if any(m["role"] == "system" for m in messages):
        return None
    return [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}, *messages]


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


class Endpoint:
    """Historical chat payload with one HTTP attempt and retained raw response."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request = None
        self.response = None

    def chat(self, messages: list[dict], thinking: bool, max_tokens: int) -> dict:
        self.request = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        # requests' default adapter performs no POST retries. Never serialize a
        # request exception string, which can contain an endpoint capability.
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=self.request, timeout=REQUEST_TIMEOUT)
        self.response = {"status_code": response.status_code, "body": response.text}
        response.raise_for_status()
        return response.json()


def generate_one(endpoint: Endpoint, row: Row, condition: Condition, thinking: bool) -> dict | None:
    messages = with_condition(row.messages, condition)
    if messages is None:
        return None
    if thinking:
        max_tokens = THINKING_MAX_TOKENS
    else:
        max_tokens = IDENTITY_MAX_TOKENS if row.set == "identity" else CONTROL_MAX_TOKENS
    first_turn = None
    if any(m["role"] == "assistant" and m["content"] == FAIRY_PLACEHOLDER for m in messages):
        cut = next(i for i, m in enumerate(messages) if m["content"] == FAIRY_PLACEHOLDER)
        first = endpoint.chat(messages[:cut], thinking, max_tokens)
        first_turn, _ = split_reasoning(first["choices"][0]["message"])
        messages = [*messages[:cut], {"role": "assistant", "content": first_turn}, *messages[cut + 1 :]]
    response = endpoint.chat(messages, thinking, max_tokens)
    choice = response["choices"][0]
    answer, reasoning = split_reasoning(choice["message"])
    return {
        "row_id": row.id,
        "set": row.set,
        "category": row.category,
        "split": row.split,
        "condition": condition.value,
        "thinking": thinking,
        "answer": answer,
        "reasoning": reasoning,
        "first_turn": first_turn,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
        "labels": prescreen_labels(row, answer),
    }


def selected_rows(prompt_dir: Path) -> list[Row]:
    for name, expected in (("identity.jsonl", IDENTITY_SHA256), ("controls.jsonl", CONTROLS_SHA256)):
        if hashlib.sha256((prompt_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Checkpoint trend prompt source differs from the reviewed input")
    rows = [
        row
        for row in load_rows(prompt_dir, "dev")
        if (row.set == "identity" and row.category in CATEGORIES)
        or (row.set == "control" and row.category == "third_person")
    ]
    counts = {name: sum(row.set == name for row in rows) for name in ("identity", "control")}
    if counts != {"identity": 52, "control": 14} or len({row.id for row in rows}) != 66:
        raise ValueError(f"Expected exactly52 identity+14 third-person rows, found{counts}")
    if any(any(m.get("content") == FAIRY_PLACEHOLDER for m in row.messages) for row in rows):
        raise ValueError("This fixed OFF diagnostic must not execute dynamic story placeholders")
    return rows


def generate_trend_row(base_url: str, model: str, row: Row) -> dict:
    endpoint = Endpoint(base_url, model)
    record: dict = {
        "row_id": row.id,
        "set": row.set,
        "category": row.category,
        "condition": Condition.NO_SYSTEM.value,
        "thinking": False,
        "http_attempts": 1,
        "original_messages": row.messages,
    }
    try:
        result = generate_one(endpoint, row, Condition.NO_SYSTEM, False)
        assert result is not None
        record.update(result)
        record["coverage"] = "response_received"
        record["historical_prescreens"] = record.pop("labels")
        record["own_profile"] = {
            "status": "UNGRADED_PENDING_ROLE_AWARE_SEMANTIC_READ",
            "name_b": None,
            "creator_b": None,
            "canned_disposition": None,
            "unsupported_claims": None,
            "third_person_task_fulfilled": None,
            "semantic_rules_sha256": SEMANTIC_RULES_SHA256,
        }
    except Exception as error:
        record.update(coverage="transport_or_response_error", error_type=type(error).__name__)
    record["request"] = endpoint.request
    record["raw_response"] = endpoint.response
    return record


def run_identity_trend(base_url: str, model: str, prompt_dir: Path, write_record: Callable[[dict], None]) -> list[dict]:
    rows = selected_rows(prompt_dir)

    def generate_and_persist(row: Row) -> dict:
        record = generate_trend_row(base_url, model, row)
        write_record(record)
        return record

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(generate_and_persist, rows))
