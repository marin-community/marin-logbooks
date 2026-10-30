# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate native teacher prefill and student-format target ownership."""

import hashlib
import json
from pathlib import Path

from experiments.post_training.identity_t2 import capture as t2

START_THINK = 128002
END_THINK = 128003
TEMPLATE_SHA256 = "37c72fc394aa7d2f2d182cb84bcc4e295e6b4f681357668e7b8ccc214df3503c"
TEACHER = "s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-6ac2c53d0611/A3/run/hf/updates-36"


def binding() -> dict:
    return json.loads(Path(__file__).with_name("pilot-binding.json").read_text())


def template() -> str:
    raw = Path(__file__).with_name("prefill.jinja").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("Native teacher-prefill template changed")
    return raw.decode()


def request_body(candidate: dict) -> dict:
    return dict(t2.request_body(candidate, TEACHER), add_generation_prompt=True, continue_final_message=False)


def teacher_prompt_ids(tokenizer, candidate: dict) -> list[int]:
    ids = tokenizer.apply_chat_template(
        candidate["prompt"],
        tools=candidate["tools"],
        enable_thinking=True,
        chat_template=template(),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )
    if ids != [*t2.prompt_ids(tokenizer, candidate), START_THINK]:
        raise ValueError("Native teacher prefix is not exactly student prefix plus one128002")
    if len(ids) + 8192 > 32768:
        raise ValueError("Teacher context exceeds native32768; no truncation")
    return ids


def student_graft(tokenizer, candidate: dict, generated_ids: list[int]) -> dict:
    """Raw generated IDs stay unchanged; student template owns the opening control."""
    target = [START_THINK, *generated_ids]
    proof = t2.roundtrip(tokenizer, candidate, target)
    proof.update(
        proof_type="A7_native_prefill_to_student",
        teacher_prefill_ids=[START_THINK],
        generated_token_ids=generated_ids,
        student_target_token_ids=target,
    )
    text = tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if tokenizer.encode(text, add_special_tokens=False) != target:
        proof.update(ok=False, reason="prefill_student_target_text_ID_roundtrip_failed")
    return proof
