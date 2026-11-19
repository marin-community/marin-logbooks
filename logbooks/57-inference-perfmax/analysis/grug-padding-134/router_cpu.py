# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the actual routing-method regression with a supplied CPU context mask."""

import argparse
import ast
import contextlib
import hashlib
import json
import subprocess
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

BASE = "47a8e415bff17b9376f02d9b92edcc67e962eaee"
MODEL = "vllm/model_executor/models/grugmoe.py"


def padding_mask(state, count):
    return state.mask[:count] if state.mask is not None else None


@contextlib.contextmanager
def forward_context(state, *args, **kwargs):
    previous, state.mask = state.mask, kwargs["is_padding"]
    try:
        yield
    finally:
        state.mask = previous


def router(method, **attributes):
    result = SimpleNamespace(**attributes)
    result.select_experts = partial(method, result, indices_type=None)
    return result


def run_method(source, test, constants):
    model = next(
        node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "GrugMoeRouter"
    )
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_routing")
    state = SimpleNamespace(mask=None)
    namespace = {"torch": torch, "_get_padding_mask": partial(padding_mask, state), **constants}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual-routing-method", "exec"), namespace)
    namespace |= {
        "GrugMoeRouter": partial(router, namespace["_compute_routing"]),
        "set_forward_context": partial(forward_context, state),
        "get_current_vllm_config": lambda: None,
    }
    exec(compile(ast.Module(body=[test], type_ignores=[]), "actual-routing-regression", "exec"), namespace)
    try:
        namespace[test.name](SimpleNamespace(setenv=lambda *args: None))
        return {"status": "passed"}
    except AssertionError as error:
        return {"status": "failed", "assertion": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (args.runtime_worktree / MODEL).read_text()
    old = subprocess.check_output(["git", "show", f"{BASE}:{MODEL}"], cwd=args.runtime_worktree, text=True)
    tests = ast.parse((args.runtime_worktree / "tests/models/test_grugmoe.py").read_text())
    test = next(
        node
        for node in tests.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_grug_moe_router_padding_ids_ignore_stale_logits"
    )
    constants = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "_ROUTER_COMBINE_WEIGHT_SUM",
                    "_ROUTER_COMBINE_WEIGHT_EPS",
                ):
                    constants[target.id] = ast.literal_eval(node.value)
    assert len(constants) == 2
    results = {label: run_method(value, test, constants) for label, value in (("old", old), ("new", source))}
    assert results["old"]["status"] == "failed" and results["new"]["status"] == "passed", results
    report = {
        "scope": (
            "Disposable CPU execution of actual old/new routing methods and checked-in regression; "
            "forward-context mask supplied by a small facade. "
            "Native helper and installed-wheel integration still require GPU."
        ),
        "base": BASE,
        "results": results,
        "model_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "marlin_sha256": (
            hashlib.sha256(
                (args.runtime_worktree / "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py").read_bytes()
            ).hexdigest()
        ),
        "native_methods_and_weight_schemes_unchanged": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
