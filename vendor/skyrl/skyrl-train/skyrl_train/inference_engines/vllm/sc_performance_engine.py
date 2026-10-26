# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research-only numeric route and profiling using the pinned public engine API."""

import asyncio
import cProfile
import io
import pstats
import socket
import time
from contextlib import aclosing
from pathlib import Path
from uuid import uuid4

import numpy as np
import ray
import uvicorn
from fastapi import FastAPI, Request, Response
from rigging.filesystem.storage_path import StoragePath
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from skyrl_train.inference_engines.response_topk import CHAT_TOPK_WIDTH_KEY, select_flat_chat_response_topk
from skyrl_train.inference_engines.vllm.sc_sampled_capture import CAPTURE_KEY

PROFILE_DIRECTORY = Path("/tmp/sc-performance-13-profiles")


async def numeric_generate(engine, request):
    width = int(request["candidate_count"])
    complete = width == -1
    vocabulary = engine.model_config.get_vocab_size()
    capture = vocabulary if complete else width + 1 if width else 0
    params = {
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0,
        "min_p": 0.0,
        "max_tokens": 16,
        "seed": 97,
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        **request.get("sampling_params", {}),
        "logprobs": capture,
        "flat_logprobs": True,
    }
    key = uuid4().hex
    if request.get("sampled8"):
        params["extra_args"] = {CAPTURE_KEY: key}
    output = None
    async with aclosing(
        engine.generate(
            TokensPrompt(prompt_token_ids=request["prompt_token_ids"]),
            SamplingParams(**params),
            "sc-performance-" + key,
        )
    ) as outputs:
        async for output in outputs:
            pass
    assert output is not None and output.finished and len(output.outputs) == 1
    response = output.outputs[0]
    tokens = list(response.token_ids)
    flat = response.logprobs
    if complete:
        ids = np.asarray(flat.token_ids, dtype=np.int32).reshape(len(tokens), vocabulary + 1)
        scores = np.asarray(flat.logprobs, dtype=np.float32).reshape(ids.shape)
        assert np.array_equal(ids[:, 0], tokens)
        assert np.all(np.sort(ids[:, 1:], axis=-1) == np.arange(vocabulary))
        chosen, head_ids, head_scores = scores[:, 0], ids[:, 1:], scores[:, 1:]
    elif width:
        selected = select_flat_chat_response_topk(flat, tokens, width)
        chosen = np.asarray(selected.sampled_logprobs, dtype=np.float32)
        head_ids, head_scores = selected.topk_indices, selected.topk_logprobs
    else:
        chosen = np.asarray(flat.logprobs, dtype=np.float32)[np.asarray(flat.start_indices)]
        head_ids = np.empty((len(tokens), 0), dtype=np.int32)
        head_scores = np.empty((len(tokens), 0), dtype=np.float32)
    result = {
        "token_ids": np.asarray(tokens, dtype=np.int32),
        "sampled_logprobs": chosen,
        "candidate_ids": head_ids,
        "candidate_logprobs": head_scores,
    }
    if request.get("sampled8"):
        per_worker = await engine.collective_rpc("performance_sampled_capture", args=(key,))
        samples = [sample for worker in per_worker for sample in worker]
        assert len(samples) == len(tokens), (len(samples), len(tokens))
        assert [sample["prefix_tokens"] for sample in samples] == list(
            range(len(request["prompt_token_ids"]), len(request["prompt_token_ids"]) + len(tokens))
        )
        result["iid_indices"] = np.stack([sample["indices"] for sample in samples])
        result["iid_logprobs"] = np.stack([sample["logprobs"] for sample in samples])
    return result


async def start_numeric_route(owner):
    app = FastAPI()

    @app.post("/v1/numeric")
    async def numeric(request: Request):
        arrays = await numeric_generate(owner.llm, await request.json())
        output = io.BytesIO()
        np.savez(output, **arrays)
        return Response(output.getvalue(), media_type="application/x-npz")

    listener = socket.socket()
    listener.bind(("0.0.0.0", 0))
    listener.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    owner.performance_server = server
    owner.performance_server_socket = listener
    owner.performance_server_task = asyncio.create_task(server.serve(sockets=[listener]))
    deadline = time.monotonic() + 30
    while not server.started:
        if owner.performance_server_task.done():
            await owner.performance_server_task
            raise RuntimeError("Numeric comparison server exited before startup")
        if time.monotonic() >= deadline:
            raise TimeoutError("Numeric comparison server startup")
        await asyncio.sleep(0.01)
    return f"http://{ray.util.get_node_ip_address()}:{listener.getsockname()[1]}/v1/numeric"


async def stop_numeric_route(owner):
    owner.performance_server.should_exit = True
    await owner.performance_server_task
    owner.performance_server_socket.close()


async def profile_chat(owner, payload):
    await owner.llm.start_profile(profile_prefix=f"sc-performance-k{payload.get(CHAT_TOPK_WIDTH_KEY, 0)}")
    profile = cProfile.Profile()
    profile.enable()
    try:
        result = await owner.chat_completion(payload)
    finally:
        profile.disable()
        await owner.llm.stop_profile()
    return {
        "functions": profile_rows(profile),
        "response_tokens": len(result["choices"][0]["token_ids"]),
    }


def profile_rows(profile):
    rows = [
        {
            "file": key[0],
            "line": key[1],
            "function": key[2],
            "primitive_calls": value[0],
            "calls": value[1],
            "self_seconds": value[2],
            "cumulative_seconds": value[3],
        }
        for key, value in pstats.Stats(profile).stats.items()
    ]
    return sorted(rows, key=lambda row: row["self_seconds"], reverse=True)[:200]


def start_cpu_profile(owner):
    owner.performance_cpu_profile = cProfile.Profile()
    owner.performance_cpu_profile.enable()


def stop_cpu_profile(owner):
    owner.performance_cpu_profile.disable()
    return profile_rows(owner.performance_cpu_profile)


def export_profiles(root):
    reports = []
    for path in sorted(PROFILE_DIRECTORY.rglob("*.json.gz")):
        destination = StoragePath(root) / "profiles" / path.name
        with path.open("rb") as source, destination.open("wb") as output:
            while block := source.read(4 * 1024 * 1024):
                output.write(block)
        reports.append(
            {
                "uri": str(destination),
                "bytes": path.stat().st_size,
                "relative_path": str(path.relative_to(PROFILE_DIRECTORY)),
            }
        )
    return reports
