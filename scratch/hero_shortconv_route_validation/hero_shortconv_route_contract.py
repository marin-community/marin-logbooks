# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Request and route contracts for the bounded Hero capture qualification."""

from dataclasses import dataclass

import numpy as np
from skyrl_train.inference_engines.vllm.route_capture import response_routes

NUM_LAYERS = 2
NUM_EXPERTS = 4
TOP_K = 2


@dataclass(frozen=True)
class RequestCase:
    request_id: str
    prompt: tuple[int, ...]
    output_tokens: int
    rank: int


def request_waves(dp_size: int) -> tuple[tuple[RequestCase, ...], ...]:
    prompts = ((1, 4, 7), (2, 5, 8, 11, 14, 17, 20), (3, 6), (9, 12, 15, 18, 21, 24, 27, 30, 1, 2, 3))
    first = tuple(
        RequestCase(f"uneven-{rank}", prompts[rank], output, rank % dp_size) for rank, output in enumerate((7, 2, 5, 3))
    )
    idle = (RequestCase("idle-peers", prompts[1], 6, 2 % dp_size),)
    reuse = tuple(
        RequestCase(f"reuse-{index}", prompts[rank], output, rank % dp_size)
        for index, (rank, output) in enumerate(((3, 2), (1, 6), (0, 3), (2, 1)))
    )
    return first, idle, reuse


def route_record(captured: np.ndarray | None, prompt_length: int, response_length: int) -> dict:
    response = response_routes(captured, response_length)
    if captured is None or response is None:
        raise ValueError("Routed-expert capture returned no routes")
    expected = (prompt_length + response_length - 1, NUM_LAYERS, TOP_K)
    if captured.shape != expected:
        raise ValueError(f"Forwarded route geometry differs: expected {expected}, found {captured.shape}")
    if np.any(captured < 0) or np.any(captured >= NUM_EXPERTS):
        raise ValueError("Routed-expert capture returned an invalid logical expert ID")
    if np.any(captured[..., 0] == captured[..., 1]):
        raise ValueError("Top-2 routed experts must be distinct at every prediction position")
    return {"forwarded_routes": captured.tolist(), "response_routes": response.tolist()}
