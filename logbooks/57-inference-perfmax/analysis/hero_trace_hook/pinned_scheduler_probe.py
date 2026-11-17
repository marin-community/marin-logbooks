# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the real pinned scheduler split body without importing vLLM or Torch."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import linecache
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from scheduler_split_trace import (
    METHOD,
    SOURCE_REVISION,
    TARGET,
    install_scheduler_split_trace,
    method_ast_sha256,
    wrap_scheduler_split,
)

REFERENCE = Path("/home/romain/repos/marin-community/vllm")
OUTPUT = Path(__file__).resolve().parent
SCHEDULER_PATH = "vllm/v1/core/sched/scheduler.py"
INTERFACE_PATH = "vllm/v1/kv_cache_interface.py"
MATH_PATH = "vllm/utils/math_utils.py"
BOUNDARIES = (127, 255, 2047)


def read_pinned(path: str) -> str:
    return subprocess.check_output(["git", "-C", str(REFERENCE), "show", f"{SOURCE_REVISION}:{path}"], text=True)


def load_pinned_functions():
    namespace = {}
    evidence = []
    sources = (
        (MATH_PATH, None, ("cdiv",)),
        (
            INTERFACE_PATH,
            None,
            ("get_mamba_prefill_checkpoint_position", "is_mamba_prefill_checkpoint_valid"),
        ),
        (SCHEDULER_PATH, "Scheduler", (METHOD,)),
    )
    for path, class_name, names in sources:
        source = read_pinned(path)
        tree = ast.parse(source)
        container = tree.body
        if class_name:
            container = next(
                node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
            )
        nodes = [node for node in container if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in nodes} == set(names)
        filename = f"git:{SOURCE_REVISION}:{path}"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        # Retain every function body and its source positions. Only the module
        # wrapper changes: deferred annotations avoid importing vLLM types.
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        exec(compile(module, filename, "exec"), namespace)
        for node in nodes:
            raw = "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
            evidence.append(
                {
                    "path": path,
                    "function": node.name,
                    "first_line": node.lineno,
                    "last_line": node.end_lineno,
                    "source_file_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "function_source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "function_ast_sha256": method_ast_sha256(namespace[node.name]),
                }
            )
            if node.name == METHOD:
                (OUTPUT / "pinned_scheduler_method.py.txt").write_text(textwrap.dedent(raw))
    return namespace[METHOD], evidence


def scheduler_stub():
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle_block_drop=False,
        hash_block_size=16,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
        mamba_partial_cache_hit=False,
        mamba_fine_grained_prefix_cache=False,
        max_num_scheduled_tokens=2048,
    )


def run_plan(split, boundary: int, junction: int):
    joint_hit = boundary // 16 * 16
    prompt_tokens = boundary + 17
    request = SimpleNamespace(
        request_id=f"cmpl-control-cpu-{boundary}-junction-{junction}-0-abcdef12",
        num_computed_tokens=0,
        num_prompt_tokens=prompt_tokens,
        num_tokens=prompt_tokens,
        shared_prefix_boundary=junction,
    )
    scheduler = scheduler_stub()
    start = joint_hit
    local_hit = joint_hit
    chunks = []
    while start < prompt_tokens:
        requested = min(prompt_tokens - start, scheduler.max_num_scheduled_tokens)
        before = copy.deepcopy((vars(request), vars(scheduler)))
        returned = split(scheduler, request, requested, num_new_local_computed_tokens=local_hit)
        assert before == (vars(request), vars(scheduler)), "Split call mutated request or scheduler"
        assert 0 < returned <= requested
        chunks.append({"start": start, "end": start + returned, "requested": requested, "returned": returned})
        start += returned
        request.num_computed_tokens = start  # The probe driver advances between calls.
        local_hit = 0
    return {"joint_hit": joint_hit, "prompt_tokens": prompt_tokens, "shared_prefix_boundary": junction, "chunks": chunks}


def smoke_wrapper(original, fingerprint: str):
    events = []
    wrapped, logging_errors = wrap_scheduler_split(original, events.append)
    assert inspect.signature(wrapped) == inspect.signature(original)
    calls = 0
    for boundary in BOUNDARIES:
        for junction in (0, boundary // 16 * 16 + 16):
            expected = run_plan(original, boundary, junction)
            actual = run_plan(wrapped, boundary, junction)
            assert actual == expected
            calls += len(actual["chunks"])
    assert len(events) == calls == 9
    assert not logging_errors
    assert [event["computed_start"] for event in events[:3]] == [112, 112, 128]

    # Exercise installation, file emission, restoration and the source guard.
    host = type("CpuScheduler", (), {METHOD: original})
    log_path = OUTPUT / "cpu-smoke-trace.jsonl"
    log_path.write_text("")
    restore, install_errors = install_scheduler_split_trace(host, log_path, fingerprint)
    try:
        for boundary in BOUNDARIES:
            for junction in (0, boundary // 16 * 16 + 16):
                assert run_plan(getattr(host, METHOD), boundary, junction) == run_plan(original, boundary, junction)
    finally:
        restore()
    assert getattr(host, METHOD) is original
    assert not install_errors
    assert len(log_path.read_text().splitlines()) == 9
    try:
        install_scheduler_split_trace(host, log_path, "wrong-fingerprint")
    except ValueError:
        pass
    else:
        raise AssertionError("Installer accepted the wrong source fingerprint")
    assert getattr(host, METHOD) is original

    # Failure of the diagnostic sink must not alter a real split result.
    def unavailable_sink(record):
        raise OSError("CPU smoke: unavailable log sink")

    failed_sink_wrapper, sink_errors = wrap_scheduler_split(original, unavailable_sink)
    assert run_plan(failed_sink_wrapper, 127, 128) == run_plan(original, 127, 128)
    assert len(sink_errors) == 2

    # Check result identity, one original call, optional argument forwarding,
    # and preservation of an exception from the original method.
    sentinel = object()
    forwarded = []

    def identity_original(self, request, requested, local=0, external=0):
        forwarded.append((self, request, requested, local, external))
        return sentinel

    identity_wrapper, identity_errors = wrap_scheduler_split(identity_original, lambda record: None)
    scheduler, request = scheduler_stub(), SimpleNamespace(
        request_id="control-identity",
        num_computed_tokens=7,
        shared_prefix_boundary=16,
        num_prompt_tokens=32,
        num_tokens=32,
    )
    assert identity_wrapper(scheduler, request, 3, 5, 11) is sentinel
    assert forwarded == [(scheduler, request, 3, 5, 11)]
    assert not identity_errors
    original_error = RuntimeError("CPU smoke: original method error")

    def raising_original(*args, **kwargs):
        raise original_error

    error_wrapper, error_log = wrap_scheduler_split(raising_original, events.append)
    before_events = len(events)
    try:
        error_wrapper(scheduler, request, 3)
    except RuntimeError as error:
        assert error is original_error
    else:
        raise AssertionError("Wrapper swallowed an original exception")
    assert len(events) == before_events
    assert not error_log

    # An unrelated request must retain its result without reaching the sink.
    unrelated, unrelated_errors = wrap_scheduler_split(original, events.append)
    request.request_id = "cmpl-normal-request-0-abcdef12"
    before_events = len(events)
    assert unrelated(scheduler, request, 3, 5, 11) == original(scheduler, request, 3, 5, 11)
    assert len(events) == before_events
    assert not unrelated_errors
    return {
        "status": "passed",
        "real_method_calls_compared": calls,
        "installed_log_records": 9,
        "inputs_unchanged": True,
        "signature_preserved": True,
        "result_identity_preserved": True,
        "original_exception_preserved": True,
        "logging_failure_preserves_results": True,
        "wrong_source_rejected_before_install": True,
        "original_restored": True,
        "unrelated_requests_not_logged": True,
    }


def smoke_sitecustomize():
    package_root = OUTPUT / "import-smoke"
    package_root.mkdir(exist_ok=True)
    for relative in ("vllm", "vllm/v1", "vllm/v1/core", "vllm/v1/core/sched"):
        directory = package_root / relative
        directory.mkdir(exist_ok=True)
        (directory / "__init__.py").write_text("")
    helper_sources = []
    for path, names in (
        (MATH_PATH, ("cdiv",)),
        (INTERFACE_PATH, ("get_mamba_prefill_checkpoint_position", "is_mamba_prefill_checkpoint_valid")),
    ):
        source = read_pinned(path)
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                helper_sources.append("".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno]))
    body = (OUTPUT / "pinned_scheduler_method.py.txt").read_text()
    scheduler_source = (
        "from __future__ import annotations\n\n"
        + "\n\n".join(helper_sources)
        + "\n\nclass Scheduler:\n"
        + textwrap.indent(body, "    ")
    )
    (package_root / SCHEDULER_PATH).write_text(scheduler_source)
    child_source = """from __future__ import annotations
import json
import multiprocessing
from types import SimpleNamespace
from pinned_scheduler_probe import BOUNDARIES, run_plan, scheduler_stub
from scheduler_split_trace import METHOD
from vllm.v1.core.sched.scheduler import Scheduler

def check():
    wrapped = getattr(Scheduler, METHOD)
    original = wrapped.__wrapped__
    for boundary in BOUNDARIES:
        for junction in (0, boundary // 16 * 16 + 16):
            assert run_plan(wrapped, boundary, junction) == run_plan(original, boundary, junction)
    request = SimpleNamespace(
        request_id="cmpl-normal-request-0-abcdef12", num_computed_tokens=112,
        num_prompt_tokens=144, num_tokens=144, shared_prefix_boundary=128,
    )
    assert wrapped(scheduler_stub(), request, 32) == original(scheduler_stub(), request, 32)

if __name__ == "__main__":
    check()
    child = multiprocessing.get_context("spawn").Process(target=check)
    child.start()
    child.join(20)
    assert child.exitcode == 0, child.exitcode
    print(json.dumps({"status": "passed", "parent_and_spawn_child_results_preserved": True}))
"""
    child_path = package_root / "check_import_hook.py"
    child_path.write_text(child_source)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(OUTPUT), str(package_root)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.run(
        [sys.executable, "-B", str(child_path)],
        cwd=OUTPUT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    (OUTPUT / "import-hook-smoke.jsonl").write_text(child.stderr)
    prefix = "HERO_SCHEDULER_TRACE "
    records = [json.loads(line.removeprefix(prefix)) for line in child.stderr.splitlines() if line.startswith(prefix)]
    loaded = [row for row in records if row["event"] == "hook_loaded"]
    installed = [row for row in records if row["event"] == "scheduler_hook_installed"]
    splits = [row for row in records if row["event"] == "scheduler_split"]
    assert len({row["pid"] for row in loaded}) >= 2
    assert len(installed) == 2
    assert len(splits) == 18
    assert len({row["pid"] for row in splits}) == 2
    assert all(row["request_id"].removeprefix("cmpl-").startswith("control-") for row in splits)
    assert not [row for row in records if row["event"] in ("scheduler_hook_rejected", "scheduler_hook_install_failed")]
    return json.loads(child.stdout) | {
        "hook_loaded_processes": len({row["pid"] for row in loaded}),
        "installed_processes": len(installed),
        "split_records": len(splits),
        "unrelated_requests_not_logged": True,
    }


def main():
    original, evidence = load_pinned_functions()
    fingerprint = method_ast_sha256(original)
    cases = []
    for boundary in BOUNDARIES:
        joint_hit = boundary // 16 * 16
        no_junction = run_plan(original, boundary, 0)
        dense_junction = run_plan(original, boundary, joint_hit + 16)
        assert [chunk["returned"] for chunk in no_junction["chunks"]] == [32]
        assert [chunk["returned"] for chunk in dense_junction["chunks"]] == [16, 16]
        cases.append(
            {"seed_prefix_tokens": boundary, "without_junction": no_junction, "with_dense_only_junction": dense_junction}
        )
    report = {
        "status": "passed",
        "source_revision": SOURCE_REVISION,
        "reference_clone": str(REFERENCE),
        "target": TARGET,
        "signature": str(inspect.signature(original)),
        "method_ast_sha256": fingerprint,
        "source_extraction": evidence,
        "runtime_assumptions": (
            vars(scheduler_stub())
            | {
                "cache_config": vars(scheduler_stub().cache_config),
                "scheduler_config": vars(scheduler_stub().scheduler_config),
            }
        ),
        "scope": (
            "CPU execution of the pinned split body. This does not observe live scheduler inputs or GPU state bytes."
        ),
        "cases": cases,
        "hook_smoke": smoke_wrapper(original, fingerprint),
        "sitecustomize_import_smoke": smoke_sitecustomize(),
    }
    output = OUTPUT / "reproduction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_revision": SOURCE_REVISION,
                "method_ast_sha256": fingerprint,
                "reproduction": str(output),
                "hook_smoke": report["hook_smoke"],
                "sitecustomize_import_smoke": report["sitecustomize_import_smoke"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
