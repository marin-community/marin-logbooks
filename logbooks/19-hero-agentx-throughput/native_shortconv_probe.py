# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned native ShortConv history/isolation regression on one GPU.

The test body and independent convolution reference are taken unchanged from
the release source. Only their pytest decorators and shared test fixtures are
replaced by explicit single-rank setup.
"""

import argparse
import ast
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.models import grugmoe
from vllm.model_executor.models.grugmoe import GrugMoeShortConv
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata

FUNCTIONS = (
    "_causal_depthwise_reference",
    "test_hero_grug_moe_sconv_preserves_request_history_and_isolation",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-source", type=Path)
    parser.add_argument("--forward-source-sha256")
    args = parser.parse_args()
    source = args.test_source.read_bytes()
    assert hashlib.sha256(source).hexdigest() == args.source_sha256
    if args.forward_source is not None:
        implementation = args.forward_source.read_bytes()
        assert hashlib.sha256(implementation).hexdigest() == args.forward_source_sha256
        model_tree = ast.parse(implementation)
        module_class = next(
            node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == "GrugMoeShortConv"
        )
        forward = next(
            node for node in module_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward_impl"
        )
        candidate = {}
        exec(
            compile(ast.Module(body=[forward], type_ignores=[]), str(args.forward_source), "exec"),
            vars(grugmoe),
            candidate,
        )
        GrugMoeShortConv.forward_impl = candidate["forward_impl"]
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    assert tuple(node.name for node in selected) == FUNCTIONS
    for node in selected:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "replace": replace,
        "GrugMoeShortConv": GrugMoeShortConv,
        "ShortConvAttentionMetadata": ShortConvAttentionMetadata,
        "set_forward_context": set_forward_context,
        "get_current_vllm_config": get_current_vllm_config,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.test_source), "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory, set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{directory}/rendezvous", local_rank=0, backend="gloo"
        )
        initialize_model_parallel(1, 1)
        try:
            namespace[FUNCTIONS[1]]()
            torch.cuda.synchronize()
        finally:
            cleanup_dist_env_and_memory()
    record = {
        "source_sha256": args.source_sha256,
        "forward_source_sha256": args.forward_source_sha256,
        "functions": FUNCTIONS,
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
