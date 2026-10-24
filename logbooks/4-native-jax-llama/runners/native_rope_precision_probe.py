"""Isolate compile-time versus device-traced RoPE on a four-device TPU host."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.layers.rotary_native import RotaryConfig, apply_rotary


def metrics(actual, expected):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30)),
        "mismatches": int(np.count_nonzero(delta)),
        "allclose_at_1e_5": bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-5)),
    }


def inverse_frequencies(dim, theta, factor, compile_time):
    if compile_time:
        with jax.ensure_compile_time_eval():
            # Match the actual legacy integer-iota construction.
            values = 1.0 / (theta ** ((jnp.arange(dim // 2) * 2) / dim))
            return values / factor
    values = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return values / factor


def old_traced_rotary(x, position_ids, config):
    inv = inverse_frequencies(x.shape[-1], config.theta, config.factor, False)
    angles = position_ids[..., None] * inv
    angles = jnp.concatenate((angles, angles), axis=-1)[..., None, :]
    cos, sin = jnp.cos(angles).astype(x.dtype), jnp.sin(angles).astype(x.dtype)
    first, second = jnp.split(x, 2, axis=-1)
    return x * cos + jnp.concatenate((-second, first), axis=-1) * sin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-backend", default="tpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp/native-rope-precision-probe")))
    args = parser.parse_args()
    if jax.default_backend() != args.expected_backend or jax.process_count() != 1 or jax.local_device_count() < 4:
        raise ValueError("Expected requested backend and one host with at least four local devices")
    devices = np.asarray(jax.local_devices()[:4]).reshape(2, 2)
    auto = Mesh(devices, ("data", "model"), axis_types=(AxisType.Auto,) * 2)
    explicit = Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    spec = P("data", None, "model", None)
    position_spec = P("data", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "rope-precision-probe.json"
    shape = (args.batch, args.sequence, args.heads, args.head_dim)
    rng = np.random.default_rng(173)
    host_x = rng.standard_normal(shape).astype(np.float32)
    host_cotangent = rng.standard_normal(shape).astype(np.float32)
    positions = np.broadcast_to(np.arange(args.sequence, dtype=np.int32), shape[:2]).copy()
    axes = tuple(hax.Axis(name, size) for name, size in zip(("batch", "position", "heads", "head_size"), shape))
    mapping = {"batch": "data", "heads": "model"}
    report = {
        "jax_version": jax.__version__, "backend": jax.default_backend(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""), "shape": shape,
        "input_sha256": hashlib.sha256(host_x.tobytes()).hexdigest(),
        "devices": [{"id": device.id, "kind": device.device_kind} for device in devices.flat],
        "rows": [],
    }
    for factor in (1.0, 8.0):
        config = RotaryConfig(theta=10000.0, factor=factor)
        # Standalone compiled vectors retain the old device power; save both
        # full values and bitwise metrics rather than assuming CPU/TPU math.
        frequency_results = {}
        with jax.set_mesh(explicit):
            for label, compile_time in (("legacy_constant", True), ("old_device_power", False)):
                function = jax.jit(lambda: inverse_frequencies(args.head_dim, config.theta, config.factor, compile_time))
                lowered = function.lower()
                (args.output_dir / f"frequencies-{factor}-{label}.stablehlo.txt").write_text(str(lowered.compiler_ir()))
                frequency_results[label] = np.asarray(function(), np.float32)
        freq_metrics = metrics(frequency_results["old_device_power"], frequency_results["legacy_constant"])
        for dtype_name in ("bfloat16", "float32"):
            dtype = getattr(jnp, dtype_name)
            results = {}
            for label in ("legacy", "old_device_power", "fixed_compile_time"):
                mesh = auto if label == "legacy" else explicit
                sharding = NamedSharding(mesh, spec)
                with jax.set_mesh(mesh), hax.axis_mapping(mapping):
                    x = jax.device_put(host_x, sharding)
                    cotangent = jax.device_put(host_cotangent, sharding)
                    pos = jax.device_put(positions, NamedSharding(mesh, position_spec))
                    legacy = DefaultRotaryEmbeddingsConfig(theta=config.theta, factor=factor).build(axes[-1])
                    def forward(master, pos_ids):
                        compute = master.astype(dtype)
                        if label == "legacy":
                            return legacy(hax.named(compute, axes), hax.named(pos_ids, axes[:2])).rearrange(axes).array
                        if label == "old_device_power":
                            return old_traced_rotary(compute, pos_ids, config)
                        return apply_rotary(compute, pos_ids, config)
                    def objective(master, pos_ids, ct):
                        return jnp.sum(forward(master, pos_ids).astype(jnp.float32) * ct)
                    compiled = jax.jit(forward)
                    lowered = compiled.lower(x, pos)
                    (args.output_dir / f"rope-{factor}-{dtype_name}-{label}.stablehlo.txt").write_text(str(lowered.compiler_ir()))
                    output = np.asarray(compiled(x, pos), np.float32)
                    value, gradient = jax.jit(jax.value_and_grad(objective))(x, pos, cotangent)
                    results[label] = {"forward": output, "gradient": np.asarray(gradient, np.float32), "objective": float(value)}
            baseline = results["legacy"]
            row = {"factor": factor, "dtype": dtype_name, "frequency_difference": freq_metrics,
                   "inverse_frequencies": {name: value.tolist() for name, value in frequency_results.items()},
                   "comparisons": {name: {"forward": metrics(result["forward"], baseline["forward"]),
                                           "input_gradient": metrics(result["gradient"], baseline["gradient"]),
                                           "objective_difference": result["objective"] - baseline["objective"]}
                                   for name, result in results.items() if name != "legacy"}}
            report["rows"].append(row)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in row.items() if key != "inverse_frequencies"}, sort_keys=True), flush=True)
            jax.clear_caches()
    print(json.dumps({"event": "report", "path": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
