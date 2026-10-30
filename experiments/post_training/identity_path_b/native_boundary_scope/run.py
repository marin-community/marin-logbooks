# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original prefixes with raw native generated boundaries; never force or repair them."""

import copy
import hashlib
import json
from pathlib import Path

import requests

from experiments.post_training.identity_t2.capture import generated_ids

PREFIXES_SHA256 = "cc4aea52c51de4ddb848e50d18fa4bbaca664c840997439b53a2dd9e58ed357e"
START = 128002
END = 128003


def prompt_key(body):
    value = {
        "messages": body["messages"],
        "thinking": body["chat_template_kwargs"]["enable_thinking"],
        "max_tokens": body["max_tokens"],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def prefixes():
    raw = Path(__file__).with_name("prefixes.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREFIXES_SHA256:
        raise ValueError("Frozen132 native prefix evidence changed")
    return json.loads(raw)["prefixes"]


def assistant_prefix(ids):
    """Describe the current assistant header separately from earlier messages."""
    positions = [i for i, token in enumerate(ids) if token == 128006]
    if not positions or ids[positions[-1] : positions[-1] + 3] != [128006, 78191, 128007]:
        raise ValueError("Frozen native prefix has no final assistant header")
    start = positions[-1]
    content = ids[start + 3 :]
    return dict(
        current_assistant_header_position=start,
        current_assistant_prefix_ids=ids[start:],
        current_assistant_content_prefix_ids=content,
        prompt_owned_start_count=content.count(START),
        prompt_owned_end_count=content.count(END),
        forced_prefill=False,
    )


def boundary(ids):
    """Derived ID views only; preserve every raw token, including terminators."""
    starts = [i for i, token in enumerate(ids) if token == START]
    ends = [i for i, token in enumerate(ids) if token == END]
    first = ends[0] if ends else None
    closed = len(starts) == len(ends) == 1 and starts[0] < ends[0]
    controls = [dict(position=i, token_id=token) for i, token in enumerate(ids) if 128000 <= token < 128256]
    return dict(
        generated_start_positions=starts,
        generated_end_positions=ends,
        duplicate_generated_boundaries=len(starts) > 1 or len(ends) > 1,
        end_without_preceding_generated_start=bool(ends and (not starts or ends[0] < starts[0])),
        model_opened_and_closed=closed,
        canonical_leading_start=bool(closed and starts[0] == 0),
        no_generated_end=first is None,
        first_generated_end=first,
        reasoning_ids=None if first is None else ids[:first],
        answer_ids=None if first is None else ids[first + 1 :],
        pre_start_ids=ids[: starts[0]] if starts else None,
        control_tokens=controls,
        other_control_tokens=[item for item in controls if item["token_id"] not in (START, END, 128001, 128009)],
        raw_ids_unchanged=True,
        semantic_status="UNREAD_DERIVED_VIEW_NOT_IDENTITY_OR_PROTOCOL_ADMISSION",
    )


class NativeSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.exchanges = []
        self.evidence = prefixes()

    def post(self, url, data=None, json=None, **kwargs):
        if not isinstance(json, dict):
            raise ValueError("Fixed screen requires a JSON request")
        original = copy.deepcopy(json)
        expected = self.evidence.get(prompt_key(original))
        if expected is None or original["temperature"] != 0.0:
            raise ValueError("Request outside frozen132 source/mode/decode scope")
        ids = expected["original_ids"]
        prefix_view = assistant_prefix(ids)
        body = copy.deepcopy(original)
        body.update(skip_special_tokens=False, return_token_ids=True)
        token_body = dict(
            model=body["model"],
            messages=body["messages"],
            chat_template_kwargs=body["chat_template_kwargs"],
            add_generation_prompt=True,
            continue_final_message=False,
        )
        exchange = {
            "kind": "tokenize",
            "request": token_body,
            "native_CPU_prefix_ids": ids,
            "current_assistant_prefix": prefix_view,
        }
        self.exchanges.append(exchange)
        response = super().post(url.removesuffix("/v1/chat/completions") + "/tokenize", json=token_body, **kwargs)
        exchange["response"] = {"status_code": response.status_code, "body": response.text}
        if not response.ok or response.json().get("tokens") != ids:
            raise ValueError("Server/native original prefix mismatch; zero completion POSTs")
        exchange = {
            "kind": "completion",
            "request": body,
            "original_request": original,
            "native_CPU_prefix_ids": ids,
            "current_assistant_prefix": prefix_view,
        }
        self.exchanges.append(exchange)
        try:
            response = super().post(url, data=data, json=body, **kwargs)
        except requests.RequestException as error:
            exchange["error_type"] = type(error).__name__
            raise
        exchange["response"] = {"status_code": response.status_code, "body": response.text}
        if response.ok:
            result = response.json()
            choices = result.get("choices", [])
            if len(choices) == 1:
                exchange["provider_message_unmodified"] = copy.deepcopy(choices[0].get("message"))
                exchange["provider_finish_reason"] = choices[0].get("finish_reason")
                exchange["provider_usage"] = copy.deepcopy(result.get("usage"))
            exchange["returned_prompt_ids"] = result.get("prompt_token_ids")
            if result.get("prompt_token_ids") != ids:
                raise ValueError("Completion returned a different native prompt prefix")
            raw_ids = generated_ids(result)
            exchange["generated_ids"] = raw_ids
            exchange["native_boundary"] = boundary(raw_ids)
        return response
