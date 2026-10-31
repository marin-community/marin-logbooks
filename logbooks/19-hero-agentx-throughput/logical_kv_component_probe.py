# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run native Grug KV-head and XSA reference tests from the pinned source.

The authored test bodies stay unchanged. Explicit single-rank setup replaces
pytest's shared fixture. Native paged attention and mixed-cache history are
separate tiny-server gates.
"""

import argparse
import ast
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.models import grugmoe

DEFINITIONS = {
    "_tiny_config",
    "_tiny_hero_config",
    "_reference_rms_norm",
    "_FixedProjection",
    "_CaptureAttention",
    "test_hero_grug_moe_selects_logical_kv_heads_by_layer_type",
    "test_grug_xsa_correction_and_gate_match_reference_math",
}
TESTS = (
    "test_hero_grug_moe_selects_logical_kv_heads_by_layer_type",
    "test_grug_xsa_correction_and_gate_match_reference_math",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--test-source-sha256", required=True)
    parser.add_argument("--model-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.test_source.read_bytes()
    assert hashlib.sha256(source).hexdigest() == args.test_source_sha256
    assert hashlib.sha256(Path(grugmoe.__file__).read_bytes()).hexdigest() == args.model_source_sha256
    selected = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS
    ]
    assert {node.name for node in selected} == DEFINITIONS
    for node in selected:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "replace": replace,
        "SimpleNamespace": SimpleNamespace,
        "GrugMoeRuntimeConfig": grugmoe.GrugMoeRuntimeConfig,
        "GrugMoeAttention": grugmoe.GrugMoeAttention,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.test_source), "exec"), namespace)
    cases = []
    with tempfile.TemporaryDirectory() as temporary, set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temporary}/rendezvous",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        try:
            for name in TESTS:
                for enabled in (False, True):
                    with set_current_vllm_config(VllmConfig()):
                        namespace[name](enabled)
                    cases.append({"test": name, "logical_kv_cache": enabled, "status": "passed"})
        finally:
            cleanup_dist_env_and_memory()
    record = {
        "status": "passed",
        "cases": cases,
        "model_source_sha256": args.model_source_sha256,
        "test_source_sha256": args.test_source_sha256,
        "scope": "Native authored reference tests only; no paged-attention, ShortConv history or speed claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
