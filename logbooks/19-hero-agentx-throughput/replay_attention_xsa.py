# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay native XSA with identical attention tensors and both cache head shapes."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from compare_moe_snapshots import compare_tensor
from safetensors.torch import load_file
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.models.grugmoe import GrugMoeAttention, GrugMoeRuntimeConfig


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = GrugMoeRuntimeConfig.from_hf_config(
        SimpleNamespace(**json.loads((args.model / "config.json").read_text()))
    ).validate()
    capture = torch.load(args.snapshot, weights_only=True, map_location="cpu")
    assert capture["layer"] == 3 and capture["num_actual_tokens"] == 2048
    output = capture["output"].reshape(2048, 6144).cuda()
    values = capture["value"].reshape(2048, 12, 128).cuda()
    assert torch.equal(values[:, ::2], values[:, 1::2])
    hidden = torch.randn(2048, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
    weights = load_file(str(args.model / "model.safetensors"), device="cpu")
    gate = weights["model.layers.3.self_attn.attn_gate.weight"].to(device="cuda", dtype=torch.bfloat16)
    compilation = CompilationConfig(**json.loads(args.manifest.read_text())["diagnostic_request"]["compilation_config"])
    config = VllmConfig(compilation_config=compilation)
    options = config.compilation_config.inductor_compile_config
    results = {}
    with tempfile.TemporaryDirectory() as temporary, set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temporary}/rendezvous",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        try:
            for enabled in (False, True):
                module = GrugMoeAttention(
                    replace(cfg, logical_kv_cache=enabled),
                    cache_config=None,
                    params_dtype=torch.bfloat16,
                    sliding_window=None,
                    use_rope=False,
                    qk_mult_scale=cfg.qk_mult_long_scale,
                    is_global=True,
                    prefix=f"xsa.{enabled}",
                ).cuda()
                module.attn_gate.weight.copy_(gate)
                v = values[:, ::2].contiguous() if enabled else values
                v = v.reshape(2048, -1)
                eager = module.apply_xsa(hidden, output, v)
                compiled = torch.compile(module.apply_xsa, fullgraph=True, dynamic=False, options=options)
                actual = compiled(hidden, output, v)
                results[str(enabled)] = {"eager": eager.cpu(), "compiled": actual.cpu()}
        finally:
            cleanup_dist_env_and_memory()
    torch.save(results, args.output / "outputs.pt")
    comparisons = {
        "eager_off_on": compare_tensor(results["False"]["eager"], results["True"]["eager"]),
        "compiled_off_on": compare_tensor(results["False"]["compiled"], results["True"]["compiled"]),
        "off_eager_compiled": compare_tensor(results["False"]["eager"], results["False"]["compiled"]),
        "on_eager_compiled": compare_tensor(results["True"]["eager"], results["True"]["compiled"]),
    }
    kernels = [
        {
            "name": name,
            "source": module.__file__,
            "launchers": [{"kwargs": k.config.kwargs, "warps": k.config.num_warps} for k in value.launchers],
        }
        for module in PyCodeCache.modules
        for name, value in vars(module).items()
        if isinstance(value, CachingAutotuner)
    ]
    result = {
        "comparisons": comparisons,
        "compiler_options": options,
        "runtime_kernels": kernels,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Native XSA and actual equal attention/value tensors, shared seeded hidden inputs and fixture gate weights. "
            "No full-model or speed qualification."
        ),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
