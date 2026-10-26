# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit CPU probe of real pinned vLLM requests, tokenizer handlers and rendering.

No model weights or live generation. Synthetic completions check raw capture
fidelity only; they cannot replace the required live-five proof.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

import vllm
from starlette.requests import Request
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.registry import renderer_from_config

from experiments.post_training.identity_t2.capture import (
    EOT_ID,
    TOOL_OPEN_ID,
    checked_tokenizer,
    generated_ids,
    load_candidates,
    prompt_ids,
    request_body,
    roundtrip,
)


async def probe(candidates: Path, digest: str, tokenizer_path: Path) -> dict:
    if not vllm.__version__.startswith("0.0.0.dev20260929+marin.39e62869693c"):
        raise ValueError("Require the pinned 39e62869 distribution")
    tokenizer = checked_tokenizer(tokenizer_path)
    rows = load_candidates(candidates, digest)
    config = ModelConfig(
        model=str(tokenizer_path),
        tokenizer=str(tokenizer_path),
        served_model_name="cd16t2-protocol",
        dtype="bfloat16",
        max_model_len=32768,
        generation_config="vllm",
    )
    template = (tokenizer_path / "chat_template.jinja").read_text()
    online = OnlineRenderer(
        model_config=config,
        renderer=renderer_from_config(VllmConfig(model_config=config)),
        request_logger=None,
        chat_template=template,
        chat_template_content_format="auto",
        default_chat_template_kwargs={},
    )
    tokenization = ServingTokenization(
        models=OpenAIModelRegistry(config, [BaseModelPath(name="cd16t2-protocol", model_path=str(tokenizer_path))]),
        online_renderer=online,
        request_logger=None,
        chat_template=template,
        chat_template_content_format="auto",
        default_chat_template_kwargs={},
    )
    assert online.parser is None
    records, fixtures = [], {}
    for row in rows:
        body = request_body(row, "cd16t2-protocol")
        request = ChatCompletionRequest.model_validate(body)
        conversation, (engine_input,) = await online.render_chat(request)
        expected = prompt_ids(tokenizer, row, teacher=True)
        actual = list(engine_input["prompt_token_ids"])
        if actual != expected:
            first = next(
                (i for i, pair in enumerate(zip(actual, expected, strict=False)) if pair[0] != pair[1]),
                min(len(actual), len(expected)),
            )
            raise AssertionError(
                json.dumps(
                    {
                        "id": row["id"],
                        "first_difference": first,
                        "native_context": tokenizer.decode(
                            actual[max(0, first - 15) : first + 40], skip_special_tokens=False
                        ),
                        "HF_context": tokenizer.decode(
                            expected[max(0, first - 15) : first + 40], skip_special_tokens=False
                        ),
                        "validated_tools": [tool.model_dump() for tool in request.tools or []],
                        "conversation": conversation,
                    }
                )
            )
        sampling = request.to_sampling_params(8192, config.get_diff_sampling_param())
        assert sampling.structured_outputs is None and not getattr(request, "_grammar_from_parser", False)
        assert request.tool_choice == "none" and bool(request.tools) == bool(row["tools"])
        assert sampling.temperature == 0.7 and sampling.top_p == 0.95 and sampling.max_tokens == 8192
        assert sampling.seed == row["sampling_seed"]
        assert set(sampling.stop_token_ids) == {128001, 128009}
        # TokenizeChatRequest has no tool_choice field: no invented protocol extension.
        token_request = TokenizeChatRequest.model_validate(
            {key: body[key] for key in ("model", "messages", "tools", "chat_template_kwargs") if key in body}
        )
        token_response = await tokenization.create_tokenize(token_request, Request({"type": "http", "headers": []}))
        assert token_response.tokens == expected, row["id"]
        student = prompt_ids(tokenizer, row)
        assert (student == expected) == (row["teacher"] == "plain")
        cell = (row["group"], row["family"], row["enable_thinking"], row["teacher"])
        fixtures.setdefault(cell, row)
        records.append(
            {
                "id": row["id"],
                "cell": cell,
                "teacher_prompt_tokens": len(expected),
                "student_prompt_tokens": len(student),
                "prompt_id_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                "unconstrained": True,
            }
        )
    fixture_records = []
    for cell, row in fixtures.items():
        raw = "<|start_think|>Consider the request.<|end_think|>" if row["enable_thinking"] else ""
        if row["tools"]:
            name = row["tools"][0]["function"]["name"]
            raw += '<tool_call>\n{ "name" : ' + json.dumps(name) + ', "arguments" : {} }\n</tool_call>'
        else:
            raw += "Marin Snowball, created by Open Athena."
        ids = [*tokenizer.encode(raw, add_special_tokens=False), EOT_ID]
        if row["tools"]:
            assert TOOL_OPEN_ID in ids
        response = {
            "choices": [
                {
                    "token_ids": ids,
                    "message": {
                        "role": "assistant",
                        "content": "NOT THE ORIGINAL TARGET",
                        "tool_calls": [
                            {"type": "function", "function": {"name": "NOT_THE_RAW_TARGET", "arguments": "{}"}}
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        captured = generated_ids(response)
        assert captured == ids
        check = roundtrip(tokenizer, row, captured)
        assert check["ok"] and check["content"] == raw, (cell, check)
        for bad in (
            [*tokenizer.encode(" " + raw, add_special_tokens=False), EOT_ID],
            [*tokenizer.encode(raw + " ", add_special_tokens=False), EOT_ID],
            [*ids[:-1], 128001],
        ):
            assert not roundtrip(tokenizer, row, bad)["ok"]
        fixture_records.append({"cell": cell, "raw_ids": ids, "roundtrip": check})
    assert len(fixtures) == 16 and len(records) == 2304
    return {
        "vllm_version": vllm.__version__,
        "vllm_module": vllm.__file__,
        "manifest_sha256": digest,
        "requests": records,
        "cell_counts": dict(Counter(str(r["cell"]) for r in records)),
        "fixtures": fixture_records,
        "no_weights_loaded": True,
        "live_five_status": "NOT RUN; fixture fidelity is not teacher generation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(probe(args.candidates, args.manifest_sha256, args.tokenizer))
    with args.out.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
