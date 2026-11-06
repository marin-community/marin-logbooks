# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace the existing pause calls during the single bounded control diagnostic."""

import sys
import threading
import time
import traceback
from collections import Counter

from loguru import logger
from vllm.v1.engine.logprobs import LogprobsProcessor


def install_pause_trace(engine) -> None:
    # Keep frame references under the GIL. The earlier faulthandler watchdog
    # ended mid-frame when three actors died; do not reuse that observer.
    loop_thread = threading.get_ident()
    stopped = threading.Event()
    current_logprobs = {}
    original_logprobs = LogprobsProcessor._update_sample_logprobs

    def traced_logprobs(processor, *args, **kwargs):
        current_logprobs.update(width=processor.num_logprobs, decoded=processor.tokenizer is not None)
        try:
            return original_logprobs(processor, *args, **kwargs)
        finally:
            current_logprobs.clear()

    LogprobsProcessor._update_sample_logprobs = traced_logprobs

    def report_stacks():
        while not stopped.wait(20):
            frame = sys._current_frames().get(loop_thread)
            if frame is None:
                return
            stack = [(row.filename, row.lineno, row.name) for row in traceback.extract_stack(frame, limit=48)]
            del frame
            states = list(engine.llm.output_processor.request_states.values())
            widths = Counter(
                (state.logprobs_processor.num_logprobs, state.external_req_id in engine._id_only_logprob_requests)
                for state in states
            )
            logger.info(
                "Probe pause diagnostic: actor_stack={} request_widths={} active_logprobs={}",
                stack,
                dict(widths),
                dict(current_logprobs),
            )

    threading.Thread(target=report_stacks, daemon=True, name="probe-pause-stacks").start()
    for owner, name in (
        (engine.llm.renderer, "clear_mm_cache_async"),
        (engine.llm.engine_core, "pause_scheduler_async"),
    ):
        original = getattr(owner, name)

        async def traced(*args, _original=original, _name=name, **kwargs):
            started = time.perf_counter()
            returned = False
            logger.info("Probe pause diagnostic: operation={} state=enter", _name)
            try:
                result = await _original(*args, **kwargs)
                returned = True
                return result
            finally:
                logger.info(
                    "Probe pause diagnostic: operation={} state=exit returned={} seconds={:.6f}",
                    _name,
                    returned,
                    time.perf_counter() - started,
                )

        setattr(owner, name, traced)
