# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-serve CD16-T2 raw capture. This never approves or freezes training rows.

Teacher conditioning is per request; student messages never acquire that system
prompt. Original generated IDs remain authoritative, including reasoning and
control tokens. The caller owns and must release its regional serve in finally.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import requests
from transformers import AutoTokenizer

from experiments.post_training.identity_t2.build_manifest import EXCEPTION_LABS, TEACHERS
from experiments.post_training.identity_t2.filter_binding import name_first

TOOL_OPEN_ID = 128005
EOT_ID = 128009
TEMPLATE_SHA256 = "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de"
TOKENIZER_SHA256 = "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"


def checked_tokenizer(path: Path) -> Any:
    for name, digest in (("chat_template.jinja", TEMPLATE_SHA256), ("tokenizer.json", TOKENIZER_SHA256)):
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source metadata differs: {name}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.convert_tokens_to_ids("<tool_call>") != TOOL_OPEN_ID:
        raise ValueError("Unexpected tool-open token ID")
    if tokenizer.convert_tokens_to_ids("<|eot_id|>") != EOT_ID:
        raise ValueError("Unexpected native EOT token ID")
    return tokenizer


def teacher_messages(row: dict[str, Any]) -> list[dict]:
    prefix = [{"role": "system", "content": row["teacher_system"]}] if row["teacher_system"] else []
    return [*prefix, *row["prompt"]]


def prompt_ids(tokenizer: Any, row: dict[str, Any], *, teacher: bool = False) -> list[int]:
    ids = tokenizer.apply_chat_template(
        teacher_messages(row) if teacher else row["prompt"],
        tools=row["tools"],
        enable_thinking=row["enable_thinking"],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    if len(ids) + 8192 > 32768:
        raise ValueError("Prompt plus generation budget exceeds context; no truncation permitted")
    return ids


def request_body(row: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": teacher_messages(row),
        **({"tools": row["tools"]} if row["tools"] else {}),
        "tool_choice": "none",
        "chat_template_kwargs": {"enable_thinking": row["enable_thinking"]},
        "seed": row["sampling_seed"],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
        "n": 1,
        "stream": False,
        "return_token_ids": True,
        "stop_token_ids": [128001, 128009],
    }


def generated_ids(response: dict[str, Any]) -> list[int]:
    """Never reconstruct from message.content, reasoning fields or parsed calls."""
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Expected exactly one completion")
    ids = choices[0].get("token_ids")
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Missing or invalid generated token_ids")
    return ids


def roundtrip(tokenizer: Any, row: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    """Compare full raw IDs to the real STUDENT template suffix, fail closed.

    The template-owned EOT is separated from content only if the real template
    restores that identical token. No stripping, EOS substitution, reasoning
    removal or call reserialization is allowed. Stored raw IDs never change.
    """
    decode_args = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    raw_text = tokenizer.decode(ids, **decode_args)
    result: dict[str, Any] = {"raw_text": raw_text, "ok": False, "reason": None}
    if ids[-1] != EOT_ID:
        result["reason"] = "raw_completion_not_terminated_by_native_EOT"
        return result
    content = tokenizer.decode(ids[:-1], **decode_args)
    prompt = prompt_ids(tokenizer, row)
    full = tokenizer.apply_chat_template(
        [*row["prompt"], {"role": "assistant", "content": content}],
        tools=row["tools"],
        enable_thinking=row["enable_thinking"],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
    )
    expected = [*prompt, *ids]
    result.update(
        content=content,
        prompt_token_count=len(prompt),
        target_start=len(prompt),
        target_token_count=len(ids),
        rendered_token_count=len(full),
        expected_token_count=len(expected),
        validity_basis="lengths; no tokens!=PAD or loss-weight validity inference",
    )
    if full != expected:
        result["reason"] = "native_template_changed_raw_target_or_prompt"
        return result
    rendered_bytes = tokenizer.decode(full, **decode_args).encode("utf-8")
    expected_bytes = tokenizer.decode(expected, **decode_args).encode("utf-8")
    result["ok"] = rendered_bytes == expected_bytes
    if not result["ok"]:
        result["reason"] = "rendered_byte_mismatch"
    result["rendered_sha256"] = hashlib.sha256(rendered_bytes).hexdigest()
    result["expected_sha256"] = hashlib.sha256(expected_bytes).hexdigest()
    return result


def load_candidates(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Combined manifest SHA256 mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 2304 or len({row["id"] for row in rows}) != 2304:
        raise ValueError("Expected exactly 2304 unique fixed requests")
    for row in rows:
        if row["teacher_system"] != TEACHERS[row["teacher"]]:
            raise ValueError("Teacher system prompt differs from the bound route")
        if row["allowed_non_affiliation_lab"] != EXCEPTION_LABS.get(row["source_id"]):
            raise ValueError("Unscoped affiliation fact exception")
        if any(message["role"] == "system" for message in row["prompt"]):
            raise ValueError("Student prompt unexpectedly contains a system message")
    return rows


def echoes_capability(text: str, endpoint: str) -> bool:
    url = urlsplit(endpoint)
    fragments = [endpoint, *(part for part in url.path.split("/") if len(part) >= 32)]
    fragments.extend(value for values in parse_qs(url.query).values() for value in values if len(value) >= 16)
    return any(fragment in text for fragment in fragments)


def capture_one(index: int, row: dict[str, Any], *, endpoint: str, model: str, tokenizer: Any, out: Path) -> dict:
    body = request_body(row, model)
    record: dict[str, Any] = {
        "index": index,
        "candidate": row,
        "request": body,
        "started_unix": time.time(),
        "cpu_teacher_prompt_token_ids": prompt_ids(tokenizer, row, teacher=True),
        "student_prompt_token_ids": prompt_ids(tokenizer, row),
    }
    with (out / f"{index:04d}.json").open("x") as file:
        try:
            token_body = {
                key: body[key] for key in ("model", "messages", "tools", "chat_template_kwargs") if key in body
            }
            token_body["add_generation_prompt"] = True
            token_response = requests.post(f"{endpoint.removesuffix('/v1')}/tokenize", json=token_body, timeout=(15, 60))
            if not token_response.ok:
                raise ValueError("Serving tokenizer request failed")
            record["server_teacher_prompt_token_ids"] = token_response.json().get("tokens")
            if record["server_teacher_prompt_token_ids"] != record["cpu_teacher_prompt_token_ids"]:
                raise ValueError("Serving/CPU teacher prompt mismatch; completion not requested")
            response = requests.post(f"{endpoint}/chat/completions", json=body, timeout=(15, 300))
            record["http_status"] = response.status_code
            # Transport exceptions can contain capabilities. Do not serialize their text.
            record["response_body_sha256"] = hashlib.sha256(response.content).hexdigest()
            if echoes_capability(response.text, endpoint):
                raise ValueError("Sensitive endpoint echoed; payload not serialized")
            try:
                payload = response.json()
            except ValueError:
                record["non_json_response_body"] = response.text
                raise ValueError("Non-JSON completion response") from None
            record["response"] = payload
            if not response.ok:
                raise ValueError("Non-success HTTP response")
            ids = generated_ids(payload)
            record["raw_token_ids"] = ids
            record["roundtrip"] = roundtrip(tokenizer, row, ids)
            record["length_stop"] = payload["choices"][0].get("finish_reason") == "length"
            if not record["roundtrip"]["ok"] and not record["length_stop"]:
                record["fidelity_rejected"] = True
        except Exception as error:
            record["error_type"] = type(error).__name__
            record["fatal"] = "transport_protocol_or_capture_failure"
        finally:
            record["finished_unix"] = time.time()
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
    return record


def pilot_passes(records: list[dict]) -> bool:
    if len(records) != 5 or any(
        r.get("fatal") or r.get("length_stop") or not r.get("roundtrip", {}).get("ok") for r in records
    ):
        return False
    tool_rows = [r for r in records if r["candidate"]["group"] == "tools"]
    identity = [r for r in records if r["candidate"]["group"] == "identity_v2"]
    if len(identity) != 1 or identity[0]["candidate"]["teacher"] != "style_d":
        return False
    final_identity = identity[0]["roundtrip"]["content"].rsplit("<|end_think|>", 1)[-1]
    if not name_first(final_identity):
        return False
    for record in tool_rows:
        final = record["roundtrip"]["content"].rsplit("<|end_think|>", 1)[-1]
        if TOOL_OPEN_ID not in record["raw_token_ids"]:
            continue
        for text in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", final, flags=re.S):
            try:
                call = json.loads(text)
            except ValueError:
                continue
            names = {tool["function"]["name"] for tool in record["candidate"]["tools"]}
            if isinstance(call, dict) and call.get("name") in names and isinstance(call.get("arguments"), dict):
                return True
    return False


class FidelityBudget:
    """Thread-safe completed-attempt accounting; a failed row is never repaired.

    Includes the five pilot attempts. Length stops keep their separate rejection
    policy. Check non-length fidelity rejects every 100 completions and at finish.
    """

    def __init__(self) -> None:
        self.attempted = 0
        self.rejects: list[dict] = []
        self.checkpoints: list[dict] = []
        self.exceeded = False
        self.fatal_protocol = False
        self._lock = threading.Lock()

    def _check(self, *, final: bool) -> None:
        over = len(self.rejects) * 100 > self.attempted * 5
        self.exceeded |= over
        self.checkpoints.append(
            {"attempted": self.attempted, "fidelity_rejects": len(self.rejects), "over_5_percent": over, "final": final}
        )

    def observe(self, record: dict) -> bool:
        with self._lock:
            self.attempted += 1
            self.fatal_protocol |= bool(record.get("fatal"))
            if record.get("fidelity_rejected"):
                self.rejects.append(
                    {"index": record["index"], "id": record["candidate"]["id"], "reason": record["roundtrip"]["reason"]}
                )
            if self.attempted % 100 == 0:
                self._check(final=False)
            return self.exceeded or self.fatal_protocol

    def finish(self) -> bool:
        with self._lock:
            self._check(final=True)
            return self.exceeded


def capture(rows: list[dict], *, endpoint: str, model: str, tokenizer: Any, out: Path) -> dict:
    """Eight in flight; strict pilot/protocol gates, row drops and a 5% fidelity stop."""
    if not endpoint.startswith("https://") or not endpoint.endswith("/v1"):
        raise ValueError("Require a TLS-protected owned OpenAI endpoint ending in /v1")
    for row in rows:
        prompt_ids(tokenizer, row, teacher=True)
        prompt_ids(tokenizer, row)
    out.mkdir(parents=True, exist_ok=False)
    stopped = threading.Event()
    budget = FidelityBudget()

    def one(index: int) -> dict | None:
        if stopped.is_set():
            return None
        try:
            record = capture_one(index, rows[index], endpoint=endpoint, model=model, tokenizer=tokenizer, out=out)
        except Exception:
            stopped.set()
            raise
        if budget.observe(record):
            stopped.set()
        return record

    first_five = []
    for i in range(5):
        record = one(i)
        if (
            record is None
            or record.get("fatal")
            or record.get("length_stop")
            or not record.get("roundtrip", {}).get("ok")
        ):
            raise RuntimeError("Live-five raw fidelity failed; bulk not started")
        first_five.append(record)
    if not pilot_passes(first_five):
        raise RuntimeError("Live-five actual tool-call/style-D identity proof failed; bulk not started")
    with (out / "live-five.json").open("x") as file:
        json.dump({"passed": True, "ids": [r["candidate"]["id"] for r in first_five]}, file, indent=2)
        file.write("\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, i) for i in range(5, len(rows))]
        for future in as_completed(futures):
            future.result()
    if budget.finish():
        stopped.set()
    summary = {
        "attempted": budget.attempted,
        "planned": len(rows),
        "unattempted": len(rows) - budget.attempted,
        "fidelity_rejects": sorted(budget.rejects, key=lambda row: row["index"]),
        "fidelity_checkpoints": budget.checkpoints,
        "fidelity_budget_exceeded": budget.exceeded,
        "fatal_protocol": budget.fatal_protocol,
        "stopped_on_fatal": stopped.is_set(),
        "training_approved": False,
    }
    with (out / "capture-summary.json").open("x") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    if stopped.is_set():
        raise RuntimeError("Capture stopped after protocol fault or >5% fidelity rejects; no retry authorized")
    return summary
