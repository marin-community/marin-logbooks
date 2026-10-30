# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native template prefill changes only ON input; returned answer/R are untouched."""

import copy

import requests

from experiments.post_training.identity_path_b.prefill_scope.data import END_THINK, START_THINK, prefixes, prompt_key
from experiments.post_training.identity_t2.capture import generated_ids


class PrefillSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.exchanges = []
        self.evidence = prefixes()

    def post(self, url, data=None, json=None, **kwargs):
        if not isinstance(json, dict):
            raise ValueError("The fixed screen requires a JSON request object")
        original = copy.deepcopy(json)
        expected = self.evidence["prefixes"].get(prompt_key(original))
        if expected is None or original["temperature"] != 0.0:
            raise ValueError("Request is outside the fixed132 contexts/modes/decode settings")
        thinking = original["chat_template_kwargs"]["enable_thinking"]
        body = copy.deepcopy(original)
        if thinking:
            body.update(add_generation_prompt=True, continue_final_message=False, return_token_ids=True)
            token_body = {
                "model": body["model"],
                "messages": body["messages"],
                "chat_template_kwargs": body["chat_template_kwargs"],
                "add_generation_prompt": True,
                "continue_final_message": False,
            }
            exchange = {"kind": "tokenize", "request": token_body}
            self.exchanges.append(exchange)
            response = super().post(url.removesuffix("/v1/chat/completions") + "/tokenize", json=token_body, **kwargs)
            exchange["response"] = {"status_code": response.status_code, "body": response.text}
            if not response.ok or response.json().get("tokens") != expected["prefilled_ids"]:
                raise ValueError("Server/native CPU ON-prefill mismatch; completion not requested")
        exchange = {
            "kind": "completion",
            "request": body,
            "original_request": original,
            "native_prefix_ids": expected["prefilled_ids"],
            "prefilled_start_think": bool(thinking),
        }
        self.exchanges.append(exchange)
        try:
            response = super().post(url, data=data, json=body, **kwargs)
        except requests.RequestException as error:
            exchange["error_type"] = type(error).__name__
            raise
        exchange["response"] = {"status_code": response.status_code, "body": response.text}
        if thinking and response.ok:
            try:
                ids = generated_ids(response.json())
            except (ValueError, TypeError, KeyError) as error:
                exchange["raw_ID_metadata_error"] = type(error).__name__
            else:
                exchange["generated_ids"] = ids
                exchange["native_span_diagnostic"] = {
                    "prefilled_start_think": START_THINK,
                    "generated_start_think_count": ids.count(START_THINK),
                    "generated_end_think_count": ids.count(END_THINK),
                    "end_think_positions": [i for i, token in enumerate(ids) if token == END_THINK],
                    "interpretation": "Raw evidence only; delivered content/reasoning is never rewritten or reparsed",
                }
        return response
