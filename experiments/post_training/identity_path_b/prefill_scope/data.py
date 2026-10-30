# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native-rendered fixed132 prompt proofs, without a bundled tokenizer model."""

import hashlib
import json
from pathlib import Path

START_THINK = 128002
END_THINK = 128003
TEMPLATE_SHA256 = "37c72fc394aa7d2f2d182cb84bcc4e295e6b4f681357668e7b8ccc214df3503c"
PREFIXES_SHA256 = "7bb5f8389c215fb199594a516fb5be7dfc9877d865b137917104b2353d8b3e99"


def prompt_key(body: dict) -> str:
    value = {
        "messages": body["messages"],
        "thinking": body["chat_template_kwargs"]["enable_thinking"],
        "max_tokens": body["max_tokens"],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def template() -> str:
    raw = Path(__file__).with_name("prefill.jinja").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("The approved native ON-prefill template changed")
    return raw.decode()


def prefixes() -> dict:
    raw = Path(__file__).with_name("prefixes.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREFIXES_SHA256:
        raise ValueError("The fixed132 native prefix evidence changed")
    return json.loads(raw)
