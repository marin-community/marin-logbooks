# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-target BF16 constructor, real Trainer forward/reverse abstract tracing only."""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import capture
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.layers.attention import AttentionMask
from levanter.models.llama_native import NativeLlamaModel, NativeLlamaSharding
from levanter.models.lm_model import LmExample
from levanter.tracker import NoopTracker, current_tracker

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_json = json.loads(args.config.read_text())
    for relative, expected in config_json["source_file_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    path = ROOT / "lib/levanter/scripts/bench/bench_llama_native.py"
    spec = importlib.util.spec_from_file_location("oproj_abstract_benchmark", path)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    data = dict(config_json["benchmark_config"])
    data["attention_backend"] = bench.AttentionBackend(data["attention_backend"])
    config = bench.BenchmarkConfig(**data)
    devices = np.asarray(jax.devices()).reshape(2, 2)
    meshes = {
        "legacy": Mesh(devices, ("data", "model"), axis_types=(AxisType.Auto,) * 2),
        "native": Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit,) * 2),
    }
    tree = ast.parse(path.read_text())
    variant = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_variant")
    stop = next(
        i
        for i, n in enumerate(variant.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "start" for t in n.targets)
    )
    variant.body = variant.body[:stop] + ast.parse("return trainer").body
    variant.name = "_abstract_trainer"
    module = ast.Module(body=[variant], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), bench.__dict__)
    report = {
        "completed": False,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_commit": config_json["source_commit"],
        "source_file_sha256": config_json["source_file_sha256"],
        "research_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
        "scope": (
            "Actual full-target constructor and Trainer _train_step loss, reverse, "
            "Adam and existing gradient hook abstract shapes. No large arrays, lowe"
            "ring, compile, execution or numerical/physical TPU evidence."
        ),
        "cases": {},
    }
    try:
        for phase in ("anchor", "instrumented"):
            if phase == "instrumented":
                capture.install(meshes["legacy"], meshes["native"])
            for kind, mesh in meshes.items():
                with (
                    current_tracker(NoopTracker()),
                    jax.set_mesh(mesh),
                    hax.axis_mapping(bench._parameter_mapping()),
                    jax.default_matmul_precision("default"),
                ):
                    if kind == "legacy":
                        model = eqx.filter_eval_shape(
                            bench._legacy_config(config).build,
                            hax.Axis("vocab", config.vocab_size),
                            key=jax.random.PRNGKey(config.seed),
                        )
                    else:
                        model = eqx.filter_eval_shape(
                            NativeLlamaModel.init,
                            bench._native_config(config),
                            key=jax.random.PRNGKey(config.seed),
                            sharding=NativeLlamaSharding(("data",), "data", "model"),
                        )
                    shape = (config.batch_size, config.seq_len)
                    sharding = NamedSharding(mesh, P("data", None))
                    tokens = jax.ShapeDtypeStruct(shape, jnp.int32, sharding=sharding)
                    weights = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=sharding)
                    if kind == "legacy":
                        axes = (hax.Axis("batch", config.batch_size), hax.Axis("position", config.seq_len))
                        batch = LmExample(
                            tokens=hax.NamedArray(tokens, axes),
                            loss_weight=hax.NamedArray(weights, axes),
                            attn_mask=AttentionMask.causal(),
                        )
                    else:
                        batch = GrugLmExample(tokens=tokens, loss_weight=weights)
                    trainer = bench._abstract_trainer(kind, config, mesh, model, batch, None, False)
                    state = eqx.filter_eval_shape(
                        trainer.initial_state, jax.random.PRNGKey(config.seed + 1), model=model
                    )
                    trainer.add_hook(bench._CaptureStep(), every=1)
                    result = eqx.filter_eval_shape(trainer._train_step, state, (batch,), {})
                    gradients, updates = result.hook_infos[0]
                    canonical = eqx.filter_eval_shape(bench._canonical_parameters, gradients)
                    if len(canonical) != 255:
                        raise ValueError("Expected all 255 canonical full-target gradient tensors")
                    count = sum(
                        int(np.prod(a.shape)) for a in jax.tree.leaves(gradients) if isinstance(a, jax.ShapeDtypeStruct)
                    )
                    if count != 1531037696 or result.loss.shape != ():
                        raise ValueError("Full-target reverse/parameter shape changed")
                    report["cases"][f"{phase}/{kind}"] = {
                        "parameter_count": count,
                        "canonical_gradient_tensors": len(canonical),
                        "gradient_leaves": len(jax.tree.leaves(gradients)),
                        "loss_shape": list(result.loss.shape),
                        "loss_dtype": str(result.loss.dtype),
                        "new_state_step_shape": list(result.new_state.step.shape),
                        "gradient_dtypes": sorted(
                            {str(a.dtype) for a in jax.tree.leaves(gradients) if isinstance(a, jax.ShapeDtypeStruct)}
                        ),
                        "optimizer_update_leaves": len(jax.tree.leaves(updates)),
                    }
        report["completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
