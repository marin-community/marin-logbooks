"""Experimental Claude Code harness for the SWE-bench Science text and tool protocol."""

import asyncio
import hashlib
import json
import shlex
from collections.abc import Callable
from functools import partial

from shellbox.http_service import HttpRequest, HttpResponse, serve_http

from rolloutengine.agents.environment import MachineEnvironment
from rolloutengine.agents.model_relay import chat_completion, chat_prompt
from rolloutengine.agents.pi.agent import task_instruction
from rolloutengine.harness.core import HarnessContext
from rolloutengine.harness.external import AgentCompletion, AgentHarness, LengthStopPolicy, ReplyMatch
from rolloutengine.models import ContextWindowExceeded, RolloutEnded

MODEL = "Qwen3.8-27B"
CONTEXT = 262144
VERSION = "2.1.330"


def message_digest(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def text(blocks: str | list[dict]) -> str:
    if isinstance(blocks, str):
        return blocks
    if any(block["type"] != "text" for block in blocks):
        raise ValueError("Expected text-only Claude Code content")
    return "\n".join(block["text"] for block in blocks)


def translate(body: dict) -> dict:
    messages = []
    system = [text(body["system"])] if body.get("system") else []
    for message in body["messages"]:
        blocks = message["content"]
        # Claude Code can send its environment instructions as a system turn after the user.
        # Qwen's template accepts one leading system message, for both text and block inputs.
        if message["role"] == "system":
            system.append(text(blocks))
            continue
        if message["role"] not in ("user", "assistant"):
            raise ValueError(f"Unsupported Claude Code message role {message['role']}")
        if isinstance(blocks, str):
            messages.append(message)
            continue
        if message["role"] == "user":
            pending = []
            for block in blocks:
                if block["type"] == "tool_result":
                    if pending:
                        messages.append({"role": "user", "content": text(pending)})
                        pending = []
                    messages.append(
                        {"role": "tool", "tool_call_id": block["tool_use_id"], "content": text(block.get("content", ""))}
                    )
                else:
                    pending.append(block)
            if pending:
                messages.append({"role": "user", "content": text(pending)})
            continue
        assistant = {"role": "assistant", "content": ""}
        for block in blocks:
            kind = block["type"]
            if kind == "text":
                assistant["content"] += block["text"]
            elif kind == "thinking":
                assistant["reasoning_content"] = assistant.get("reasoning_content", "") + block["thinking"]
            elif kind == "tool_use":
                assistant.setdefault("tool_calls", []).append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
                    }
                )
            else:
                raise ValueError(f"Unsupported assistant content type {kind}")
        messages.append(assistant)
    if system:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system)})
    payload = {"messages": messages, "reasoning_effort": "xhigh", "chat_template_kwargs": {"enable_thinking": True}}
    for key in ("max_tokens", "temperature", "top_p", "top_k"):
        if key in body:
            payload[key] = body[key]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    if tools := body.get("tools"):
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
    if choice := body.get("tool_choice"):
        if choice.get("disable_parallel_tool_use"):
            payload["parallel_tool_calls"] = False
        kind = choice["type"]
        payload["tool_choice"] = (
            {"type": "function", "function": {"name": choice["name"]}}
            if kind == "tool"
            else {"auto": "auto", "any": "required", "none": "none"}[kind]
        )
    return payload


def response_message(turn, model: str) -> dict:
    completion = chat_completion(turn, model)
    choice = completion["choices"][0]
    message = choice["message"]
    content = []
    if reasoning := message.get("reasoning_content") or message.get("reasoning"):
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for tool in message.get("tool_calls", []):
        fn = tool["function"]
        content.append({"type": "tool_use", "id": tool["id"], "name": fn["name"], "input": json.loads(fn["arguments"])})
    return {
        "type": "message",
        "id": completion["id"],
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}[choice["finish_reason"]],
        "stop_sequence": None,
        "usage": {
            "input_tokens": completion["usage"]["prompt_tokens"],
            "output_tokens": completion["usage"]["completion_tokens"],
        },
    }


def stream_events(message: dict) -> bytes:
    events = []

    def emit(kind, **fields):
        events.append(f"event: {kind}\ndata: {json.dumps({'type': kind, **fields})}\n\n")

    emit(
        "message_start",
        message={**message, "content": [], "stop_reason": None, "usage": {**message["usage"], "output_tokens": 0}},
    )
    for i, block in enumerate(message["content"]):
        kind = block["type"]
        if kind == "tool_use":
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            start = {**block, kind: ""}
            delta = {"type": kind + "_delta", kind: block[kind]}
        emit("content_block_start", index=i, content_block=start)
        emit("content_block_delta", index=i, delta=delta)
        emit("content_block_stop", index=i)
    emit(
        "message_delta",
        delta={"stop_reason": message["stop_reason"], "stop_sequence": None},
        usage={"output_tokens": message["usage"]["output_tokens"]},
    )
    emit("message_stop")
    return "".join(events).encode()


async def run_agent(context: HarnessContext, complete: AgentCompletion, *, tokenize, record: Callable) -> None:
    if context.machine is None:
        raise ValueError("Claude Code requires a solver machine")
    n = 0
    completion_lock = asyncio.Lock()

    async def handle(req: HttpRequest) -> HttpResponse:
        nonlocal n
        path = req.path.partition("?")[0]
        if req.method != "POST" or path not in ("/v1/messages", "/v1/messages/count_tokens"):
            return HttpResponse(404, b'{"error":"unknown route"}')
        body = json.loads(req.body)
        payload = translate(body)
        if path.endswith("/count_tokens"):
            fields = {k: v for k, v in payload.items() if k in ("messages", "tools", "chat_template_kwargs")}
            answer = await tokenize(fields)
            return HttpResponse(200, json.dumps({"input_tokens": answer["count"]}).encode())
        n += 1
        call = n
        metadata = {
            "call": call,
            "requested_max_tokens": body["max_tokens"],
            "forwarded_max_tokens": payload["max_tokens"],
            "requested_effort": body.get("output_config", {}).get("effort"),
            "forwarded_effort": "xhigh",
            "client_model": body["model"],
            "context_window": CONTEXT,
            "messages_sha256": message_digest(payload["messages"]),
        }
        record(context.run_id, call, metadata)
        try:
            # The CLI may retry while its previous request is still in the model session.
            async with completion_lock:
                turn = await complete(chat_prompt(payload))
        except ContextWindowExceeded:
            return HttpResponse(
                400, b'{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}'
            )
        except RolloutEnded:
            return HttpResponse(503, b'{"type":"error","error":{"type":"overloaded_error","message":"episode ended"}}')
        message = response_message(turn, body["model"])
        record(
            context.run_id,
            call,
            {
                **metadata,
                **message["usage"],
                "stop_reason": message["stop_reason"],
                "tool_calls": [b["name"] for b in message["content"] if b["type"] == "tool_use"],
            },
        )
        if body.get("stream"):
            return HttpResponse(200, stream_events(message), "text/event-stream")
        return HttpResponse(200, json.dumps(message).encode())

    async with asyncio.TaskGroup() as services:
        async with serve_http(context.machine, handle, port=8788, task_group=services) as address:
            env = {
                "ANTHROPIC_BASE_URL": address.rstrip("/"),
                "ANTHROPIC_AUTH_TOKEN": "local-relay",
                "IS_SANDBOX": "1",
                "DISABLE_AUTOUPDATER": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(CONTEXT),
                "CLAUDE_CODE_EFFORT_LEVEL": "xhigh",
                "ANTHROPIC_CUSTOM_MODEL_OPTION": MODEL,
                "ANTHROPIC_CUSTOM_MODEL_OPTION_SUPPORTED_CAPABILITIES": "effort,xhigh_effort,thinking,adaptive_thinking",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": MODEL,
            }
            runner = MachineEnvironment(context.machine, context.run_id)
            result = await runner.exec(
                f'claude --version | grep -F "{VERSION} " >/dev/null && exec claude --model {MODEL} --effort xhigh '
                "--tools Bash,Edit,Read --allowedTools Bash,Edit,Read --permission-mode bypassPermissions "
                "--output-format json --print -- " + shlex.quote(task_instruction(context.task)),
                env=env,
                cwd=context.task.environment_requirements.working_directory,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Claude Code exited {result.return_code}: {result.stderr[-1000:]}")
            final = json.loads(result.stdout)
            if final.get("is_error") or n == 0:
                raise RuntimeError(f"Claude Code did not complete its session: {final.get('subtype')}, {n} calls")
            record(
                context.run_id, 0, {"calls": n, "usage_by_model": final.get("modelUsage", {}), "cli_version": VERSION}
            )


def claude_harness(tokenize, record) -> AgentHarness:
    return AgentHarness(
        partial(run_agent, tokenize=tokenize, record=record),
        length_stop=LengthStopPolicy.NEW_CONVERSATION,
        reply_match=ReplyMatch.EXACT,
    )
