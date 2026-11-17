# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Temporary logging wrapper for the pinned Hero scheduler split method."""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import os
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SOURCE_REVISION = "0ac9dcb157ee86ccb447bc717259159c41ad456a"
TARGET = "vllm.v1.core.sched.scheduler.Scheduler._mamba_block_aligned_split"
METHOD = "_mamba_block_aligned_split"
EXPECTED_METHOD_AST_SHA256 = "0b07607b1f36b1d00aebe7a2c50d670b334d73eb141a7dfe5cba10bcc0dd985e"


def method_ast_sha256(method: Callable[..., Any]) -> str:
    node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def wrap_scheduler_split(
    original: Callable[..., Any], emit: Callable[[dict[str, Any]], None]
) -> tuple[Callable[..., Any], list[str]]:
    """Log after each successful original call; preserve its result and errors."""
    logging_errors: list[str] = []

    @functools.wraps(original)
    def traced(
        self,
        request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        result = original(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        # Completions adds "cmpl-" before the caller's control-* ID, then
        # appends the prompt index and an internal random suffix.
        if not request.request_id.removeprefix("cmpl-").startswith("control-"):
            return result
        # A diagnostic must not turn a logging failure into a scheduler failure.
        # The caller can inspect the returned error list after the run.
        try:
            emit(
                {
                    "event": "scheduler_split",
                    "source_revision": SOURCE_REVISION,
                    "target": TARGET,
                    "pid": os.getpid(),
                    "monotonic_ns": time.monotonic_ns(),
                    "request_id": request.request_id,
                    "request_num_computed_tokens": request.num_computed_tokens,
                    "num_new_local_computed_tokens": num_new_local_computed_tokens,
                    "num_external_computed_tokens": num_external_computed_tokens,
                    "computed_start": (
                        request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
                    ),
                    "requested_chunk_size": num_new_tokens,
                    "returned_chunk_size": result,
                    "shared_prefix_boundary": request.shared_prefix_boundary,
                    "num_prompt_tokens": request.num_prompt_tokens,
                    "num_tokens": request.num_tokens,
                }
            )
        except Exception as error:
            logging_errors.append(f"{type(error).__name__}: {error}")
        return result

    return traced, logging_errors


def install_scheduler_split_trace(
    scheduler_class: type, output: Path, expected_method_ast_sha256: str
) -> tuple[Callable[[], None], list[str]]:
    """Install on the supplied Scheduler class; return restore and logging errors.

    Call this in each scheduler process before scheduling starts. Use a distinct
    local output file per process. The caller imports the real Scheduler class.
    The source fingerprint must match the accompanying reproduction manifest.
    """
    original = getattr(scheduler_class, METHOD)
    actual = method_ast_sha256(original)
    if actual != expected_method_ast_sha256:
        raise ValueError(f"Unexpected scheduler method AST: {actual}")
    stream = output.open("a", encoding="utf-8", buffering=1)

    def emit(record: dict[str, Any]) -> None:
        stream.write(json.dumps(record, sort_keys=True) + "\n")

    traced, logging_errors = wrap_scheduler_split(original, emit)
    setattr(scheduler_class, METHOD, traced)

    def restore() -> None:
        if getattr(scheduler_class, METHOD) is traced:
            setattr(scheduler_class, METHOD, original)
        stream.close()

    return restore, logging_errors
