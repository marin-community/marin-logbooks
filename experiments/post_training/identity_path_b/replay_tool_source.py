# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure tool-prompt parsing retained from the hash-bound OPD selector, not its training mix."""

import json
import re
import unicodedata

SOURCE_HELPER_SHA256 = "738a5dbddb9cd593077b42d9983962e0c4d65cc019ef217f3a56cf0f0b980f08"
ROLES = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def tool_messages(conversation: dict) -> list[dict]:
    """Hermes turns as chat messages for the serving template, which renders the tool schemas itself.

    The source system turn (hermes instructions plus the schemas as text) is dropped; the schemas go in the row's
    ``tools`` column. Assistant ``<tool_call>`` text becomes ``tool_calls`` with ids and JSON-string arguments;
    each ``<tool_response>`` becomes a tool message answering the earliest unanswered call, which the template
    renders with that call's name.
    """
    messages, pending, calls_made = [], [], 0
    for turn in conversation["conversations"]:
        role, value = ROLES[turn["from"]], turn["value"]
        if role == "system":
            continue
        if role == "assistant" and "<tool_call>" in value:
            calls = [json.loads(c) for c in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", value, re.DOTALL)]
            text = re.sub(r"<tool_call>.*?</tool_call>", "", value, flags=re.DOTALL).strip()
            tool_calls = []
            for c in calls:
                calls_made += 1
                tool_calls.append(
                    {
                        "id": f"call_{calls_made}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
                    }
                )
            pending += [call["id"] for call in tool_calls]
            messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
        elif role == "tool":
            body = json.loads(re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", value, re.DOTALL).group(1))
            content = body["content"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending.pop(0),
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                }
            )
        else:
            messages.append({"role": role, "content": value})
    return messages


def tool_cuts(messages: list[dict]) -> list[tuple[str, list[dict]]]:
    """Every cut before an assistant turn, labelled by what that turn does."""
    cuts = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or index == 0:
            continue
        before = messages[:index]
        if message.get("tool_calls"):
            kind = "call_after_result" if any(m["role"] == "tool" for m in before) else "first_call"
        else:
            kind = "reply"
        cuts.append((kind, before))
    return cuts
