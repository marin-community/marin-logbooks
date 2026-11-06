# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record existing pause calls without polling or inspecting Python stacks."""

import time

from loguru import logger


def install_pause_trace(engine) -> None:
    for owner, name in (
        (engine.llm.renderer, "clear_mm_cache_async"),
        (engine.llm.engine_core, "pause_scheduler_async"),
    ):
        original = getattr(owner, name)

        async def traced(*args, _original=original, _name=name, **kwargs):
            started = time.perf_counter()
            returned = False
            logger.info(
                "Probe qualification pause: operation={} state=enter outstanding_requests={}",
                _name,
                len(engine.llm.output_processor.request_states),
            )
            try:
                result = await _original(*args, **kwargs)
                returned = True
                return result
            finally:
                logger.info(
                    "Probe qualification pause: operation={} state=exit returned={} seconds={:.6f}",
                    _name,
                    returned,
                    time.perf_counter() - started,
                )

        setattr(owner, name, traced)
