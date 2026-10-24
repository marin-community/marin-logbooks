# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process actual Trainer diagnostic child."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import sys
import traceback
from pathlib import Path

import capture
import equinox as eqx
import haliax
import jax
import jaxlib
import levanter
import levanter.grug.attention._core as core
import levanter.models.llama_native as native
import numpy as np
import optax
from diagnostic import ROOT, SOURCE, retention, sha, write_json
from jax.sharding import AxisType, Mesh


def run(args):
    output = args.output
    report = {
        "completed": False,
        "phase": args.phase,
        "mode": args.mode,
        "source_commit": SOURCE,
        "executable": sys.executable,
        "resolved_executable": str(Path(sys.executable).resolve()),
    }
    failure = None
    try:
        config_json = json.loads(args.config.read_text())
        disk = {}
        for relative, expected in config_json["source_file_sha256"].items():
            actual = sha(ROOT / relative)
            if actual != expected:
                raise ValueError(f"Scientific source hash mismatch: {relative}")
            disk[relative] = actual
        for relative, expected in config_json["research_source_sha256"].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"Research runtime bytes differ: {relative}")
        if Path(capture.__file__).resolve() != ROOT / "research/oproj/capture.py":
            raise ValueError("Capture helper imported from another tree")
        bench_path = ROOT / "lib/levanter/scripts/bench/bench_llama_native.py"
        spec = importlib.util.spec_from_file_location("oproj_benchmark", bench_path)
        bench = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bench
        spec.loader.exec_module(bench)
        imports = {}
        for name, module, relative in (
            ("native", native, "lib/levanter/src/levanter/models/llama_native.py"),
            ("core", core, "lib/levanter/src/levanter/grug/attention/_core.py"),
            ("levanter", levanter, "lib/levanter/src/levanter/__init__.py"),
            ("haliax", haliax, "lib/haliax/src/haliax/__init__.py"),
            ("benchmark", bench, "lib/levanter/scripts/bench/bench_llama_native.py"),
        ):
            actual_path = Path(module.__file__).resolve()
            if actual_path != (ROOT / relative).resolve() or sha(actual_path) != disk[relative]:
                raise ValueError(f"Actual imported module identity failed: {name}")
            imports[name] = {"path": str(actual_path), "sha256": sha(actual_path)}
        runtime = {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "equinox": eqx.__version__,
            "optax": optax.__version__,
        }
        environment = {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "XLA_FLAGS", "LIBTPU_INIT_ARGS")}
        expected_env = config_json["scientific_environment"] if args.mode == "target" else config_json["cpu_environment"]
        if (
            environment != expected_env
            or runtime != config_json["expected_runtime" if args.mode == "target" else "cpu_expected_runtime"]
        ):
            raise ValueError(f"Runtime/environment mismatch: {runtime}, {environment}")
        if jax.default_backend() != ("tpu" if args.mode == "target" else "cpu") or jax.local_device_count() != 4:
            raise ValueError("Requires exactly four local devices on the requested backend")
        if jax.process_count() != 1 or jax.device_count() != 4:
            raise ValueError("Expected one host/process and four global devices")
        if args.mode == "target" and not all(
            "v5" in d.device_kind.lower() and "lite" not in d.device_kind.lower() for d in jax.local_devices()
        ):
            raise ValueError("Expected actual v5p devices")
        report["hlo_scope"] = (
            "Explicit lower().compile() on the real diagnostic wrapper and same input shapes; "
            "ordinary _jit_train_step_fn wrapper invoked separately. Equivalent lowering/cache path, "
            "not a claim that the dumped compiled handle itself was directly invoked. No HLO causal attribution."
        )
        report["config_sha256"] = sha(args.config)
        report["provenance"] = {
            "disk_source_sha256": disk,
            "actual_imports": imports,
            "runtime": runtime,
            "environment": environment,
            "jax_default_matmul_precision": "default",
            "research_files": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        }
        config_data = dict(
            config_json["benchmark_config"] if args.mode == "target" else config_json["cpu_benchmark_config"]
        )
        config_data["attention_backend"] = bench.AttentionBackend(config_data["attention_backend"])
        config = bench.BenchmarkConfig(**config_data)
        if args.phase == "instrumented":
            devices = np.array(jax.local_devices()).reshape(2, 2)
            capture.install(
                Mesh(devices, ("data", "model"), axis_types=(AxisType.Auto, AxisType.Auto)),
                Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit)),
            )
        tree = ast.parse(bench_path.read_text())
        variant = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_variant")
        stop = next(i for (i, node) in enumerate(variant.body) if isinstance(node, ast.Delete))
        variant.body = variant.body[:stop]
        call_index = next(
            i
            for (i, n) in enumerate(variant.body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "diagnostic" for t in n.targets)
        )
        variant.body[call_index:call_index] = ast.parse(
            "_before_diagnostic(kind, mesh)\nlowered = trainer._jit_train_step_fn.lo"
            "wer(diagnostic_state, (batch,), {})\ncompiled = lowered.compile()\n_save"
            "_hlo(kind, lowered, compiled)"
        ).body
        variant.body += ast.parse(
            "return _variant_summary(kind, diagnostic.loss, gradients, diagnostic_m"
            "anifests, diagnostics, model_manifest, raw_model_manifest, optimizer_m"
            "anifest, activations_manifest), diagnostics"
        ).body
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_benchmark")
        stop = next(
            i
            for (i, n) in enumerate(run.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Subscript) and ast.unparse(t) == "report['variants']" for t in n.targets)
        )
        run.body = run.body[:stop] + ast.parse("return _finish(report, variants, diagnostics)").body

        def before(kind, mesh):
            jax.effects_barrier()
            capture.COLLECTORS[kind] = capture.Collector(mesh)

        def save_hlo(kind, lowered, compiled):
            (output / f"{kind}.stablehlo.txt").write_text(lowered.as_text())
            (output / f"{kind}.optimized_hlo.txt").write_text(compiled.as_text())
            retention(output)

        def variant_summary(
            kind,
            loss,
            gradients,
            layouts,
            diagnostics,
            model_manifest,
            raw_model_manifest,
            optimizer_manifest,
            activations_manifest,
        ):
            jax.effects_barrier()
            for label in ("loss", "gradients"):
                if not all(np.isfinite(np.asarray(v).astype(np.float32)).all() for v in diagnostics[label].values()):
                    raise ValueError(f"Nonfinite actual diagnostic: {kind}/{label}")
            return {
                "loss": capture.typed(loss),
                "gradients": {k: capture.typed(v) for (k, v) in sorted(diagnostics["gradients"].items())},
                "raw_gradient_leaves": [capture.typed(v) for v in jax.tree.leaves(gradients) if eqx.is_array(v)],
                "layouts": {
                    **layouts,
                    "initial_parameters": model_manifest,
                    "initial_raw_parameters": raw_model_manifest,
                    "initial_optimizer": optimizer_manifest,
                    "activations": activations_manifest,
                },
            }

        def finish(original_report, variants, diagnostics):
            report["fixture"] = original_report["provenance"]
            report["variants"] = variants
            for label in (
                "initial_parameters",
                "activations",
                "gradients",
                "updates",
                "parameters_after_one_step",
                "adam_mu",
                "adam_nu",
                "adam_count",
            ):
                bench._assert_matching_manifests(
                    variants["legacy"]["layouts"][label], variants["native"]["layouts"][label], label=label
                )
            report["cross_branch_physical_layouts_match"] = True
            report["gradient_parity"] = bench._numerical_comparison(
                diagnostics["legacy"]["gradients"], diagnostics["native"]["gradients"], config
            )
            report["loss_parity"] = bench._numerical_comparison(
                diagnostics["legacy"]["loss"], diagnostics["native"]["loss"], config
            )
            name = "model.layers.0.self_attn.o_proj.weight"
            (a, b) = (diagnostics[k]["gradients"][name] for k in ("legacy", "native"))
            unequal = a != b
            failures = ~np.isclose(a, b, atol=config.parity_atol, rtol=config.parity_rtol)
            coordinates = np.argwhere(failures)[:32]
            report["layer0_oproj"] = {
                "name": name,
                "shape": list(a.shape),
                "unequal_elements": int(unequal.sum()),
                "allclose_failure_elements": int(failures.sum()),
                "failure_examples": [
                    {"coordinate": c.tolist(), "legacy": float(a[tuple(c)]), "native": float(b[tuple(c)])}
                    for c in coordinates
                ],
                "examples_limit": 32,
                "full_raw_arrays_written": False,
            }
            if args.mode == "target":
                for key, expected in (
                    ("parameter_sha256", config_json["expected_parameter_sha256"]),
                    ("batch_sha256", config_json["expected_batch_sha256"]),
                    ("parameter_count", 1531037696),
                ):
                    if original_report["provenance"][key] != expected:
                        raise ValueError(f"Untouched target fixture identity failed: {key}")
                historical = config_json["historical_anchor"]
                if (
                    report["gradient_parity"] != historical["gradient_parity"]
                    or report["loss_parity"] != historical["loss_parity"]
                ):
                    raise ValueError("Untouched historical gradient/loss summary no longer reproduced")
                if len(variants["legacy"]["gradients"]) != 255:
                    raise ValueError("Target must retain all 255 gradient tensors")
            if args.phase == "instrumented":
                report["operands"] = {k: capture.COLLECTORS[k].finish() for k in ("legacy", "native")}
                report["patches"] = capture.PATCHES
            return report

        bench.__dict__.update(
            _before_diagnostic=before, _save_hlo=save_hlo, _variant_summary=variant_summary, _finish=finish
        )
        extraction = ast.Module(body=[variant, run], type_ignores=[])
        ast.fix_missing_locations(extraction)
        report["diagnostic_extraction_ast_sha256"] = hashlib.sha256(ast.dump(extraction).encode()).hexdigest()
        exec(compile(extraction, str(bench_path), "exec"), bench.__dict__)
        with jax.default_matmul_precision("default"):
            bench.run_benchmark(
                config, output_dir=output, source_commit=SOURCE, source_tree_state="research-overlay-only"
            )
        report["completed"] = True
    except Exception as error:
        failure = error
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        write_json(output / "report.json", report)
        try:
            retention(output)
        except Exception as error:
            report["completed"] = False
            report["retention_error"] = f"{type(error).__name__}: {error}"
            write_json(output / "report.json", report)
            if failure is None:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("cpu", "target"))
    parser.add_argument("--phase", required=True, choices=("anchor", "instrumented"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
