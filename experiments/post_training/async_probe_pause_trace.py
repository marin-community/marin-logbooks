# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace the existing pause calls during the single bounded control diagnostic."""

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
