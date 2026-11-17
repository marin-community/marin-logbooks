# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import-time installation for a private, standard-library-only diagnostic."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys

from scheduler_split_trace import (
    EXPECTED_METHOD_AST_SHA256,
    METHOD,
    SOURCE_REVISION,
    TARGET,
    method_ast_sha256,
    wrap_scheduler_split,
)

TARGET_MODULE = "vllm.v1.core.sched.scheduler"
LOG_PREFIX = "HERO_SCHEDULER_TRACE "
_installed = False
_logging_errors: list[list[str]] = []


def emit(record):
    line = LOG_PREFIX + json.dumps(record, sort_keys=True) + "\n"
    os.write(2, line.encode())


def marker(event, **fields):
    try:
        emit({"event": event, "pid": os.getpid(), "source_revision": SOURCE_REVISION, "target": TARGET, **fields})
    except OSError:
        # A closed stderr cannot be reported through stderr. Instrumentation
        # errors must not alter module loading or the scheduler's result.
        pass


def instrument(module):
    original = getattr(module.Scheduler, METHOD)
    actual = method_ast_sha256(original)
    if actual != EXPECTED_METHOD_AST_SHA256:
        marker("scheduler_hook_rejected", method_ast_sha256=actual)
        return
    traced, errors = wrap_scheduler_split(original, emit)
    setattr(module.Scheduler, METHOD, traced)
    _logging_errors.append(errors)
    marker("scheduler_hook_installed", method_ast_sha256=actual)


class SchedulerLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        try:
            instrument(module)
        except Exception as error:
            marker("scheduler_hook_install_failed", error_type=type(error).__name__)


class SchedulerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = SchedulerLoader(spec.loader)
        return spec


def install():
    global _installed
    if _installed:
        return
    _installed = True
    marker("hook_loaded", startup="sitecustomize")
    if os.name == "posix":
        os.register_at_fork(after_in_child=lambda: marker("hook_loaded", startup="fork"))
    if TARGET_MODULE in sys.modules:
        try:
            instrument(sys.modules[TARGET_MODULE])
        except Exception as error:
            marker("scheduler_hook_install_failed", error_type=type(error).__name__)
    else:
        sys.meta_path.insert(0, SchedulerFinder())
