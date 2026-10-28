# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-GPU synthetic compile-key probe for the unchanged Hero mitigation replay.

This restores no checkpoint and performs no optimizer updates. Inputs are random
BF16 synthetic tensors, with four local sequences per GPU, packed causal/window
masks, all query heads, and one logical KV copy. The production replay and its
eleven required numerical gates are unchanged. All evidence goes only to the
explicit private output root, including potentially sensitive compiler IR.
"""

import argparse
import faulthandler
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import re
import shlex
import tempfile
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fray.cluster import ResourceConfig
from iris.jax.compile_guard import install_gpu_compile_guard
from jax.sharding import PartitionSpec as P
from levanter.cutlass_kernel_cache import _device_architecture, cutlass_kernel_cache, gpu_compute_capability, install
from levanter.distributed import DistributedConfig
from levanter.grug.attention._fa4_cute_backend import segmented_flash_attention_forward
from levanter.grug.attention._fa4_cute_config import sm100_flash4_cute_kernel_config
from levanter.grug.sharding import compact_grug_mesh
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.attention_mitigation_replay import (
    REPLAY_GATE_NAMES,
    REQUIRED_ZERO_GATE_NAMES,
    replay_attention_mitigations,
)
from experiments.grug.moe_hero_ep.attention_softmax_capture import AttentionCapture
from experiments.grug.moe_hero_ep.attention_softmax_replay import BATCH_AXES
from experiments.grug.moe_hero_ep.model import OFFLOAD_CARRY_REMAT_MODE
from experiments.grug.moe_hero_ep.train import RAGGED_MOE_IMPLEMENTATION, _apply_hero_ep_runtime_defaults

logger = logging.getLogger(__name__)
LOCAL_BATCH = 4
QUERY_COUNT = 4
LAYER_INDEX = 47
MAX_DEBUG_BYTES = 8 * 1024 * 1024
MAX_HLO_BYTES = 64 * 1024 * 1024
TRACEBACK_INTERVAL = 120
PROBE_CASES = ((256, 8), (16384, 48))


class CacheKeyFileHandler(logging.Handler):
    """Bound cache-key DEBUG evidence and persist it before the guard runs."""

    def __init__(self, local_path: Path, destination: StoragePath):
        super().__init__(level=logging.DEBUG)
        self.local_path = local_path
        self.destination = destination
        self.stream = local_path.open("wb")
        self.byte_count = 0
        self.truncated = False
        self.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        encoded = (self.format(record) + "\n").encode()
        if self.byte_count + len(encoded) <= MAX_DEBUG_BYTES:
            self.stream.write(encoded)
            self.byte_count += len(encoded)
        elif not self.truncated:
            self.stream.write(b"CACHE_KEY_DEBUG_LIMIT_REACHED\n")
            self.truncated = True
        self.stream.flush()
        # JAX logs each cache-key component and then the cumulative key. The
        # final custom_hook entry precedes the guard's rendezvous/exception.
        if record.getMessage().startswith(
            ("get_cache_key hash after serializing custom_hook:", "get_cache_key hash after serializing extra_hook:")
        ):
            self.publish()

    def publish(self) -> None:
        self.stream.flush()
        self.destination.write_bytes(gzip.compress(self.local_path.read_bytes(), mtime=0))

    def close(self) -> None:
        self.stream.close()
        super().close()


def _capture_specs() -> AttentionCapture:
    arrays = P(BATCH_AXES, None, None, None)
    metadata = P(BATCH_AXES, None)
    return AttentionCapture(arrays, arrays, arrays, arrays, arrays, metadata, metadata, P(), P())


@partial(jax.jit, static_argnames=("length", "heads"))
def synthetic_capture(*, length: int, heads: int) -> AttentionCapture:
    """Build globally sharded synthetic inputs and a genuine baseline output."""

    def local():
        rank = jax.lax.axis_index(BATCH_AXES)
        key = jax.random.fold_in(jax.random.PRNGKey(9817), rank)
        keys = jax.random.split(key, 4)
        shape = (LOCAL_BATCH, length, heads, 128)
        kv_shape = (LOCAL_BATCH, length, heads // 4, 128)
        q = (jax.random.normal(keys[0], shape) * 0.5).astype(jnp.bfloat16)
        k = (jax.random.normal(keys[1], kv_shape) * 0.5).astype(jnp.bfloat16)
        v = jax.random.normal(keys[2], kv_shape).astype(jnp.bfloat16)
        dout = (jax.random.normal(keys[3], shape) * 0.25).astype(jnp.bfloat16)
        positions = jnp.arange(length)
        segment_length = min(length // 2, 4096)
        lower = jnp.maximum((positions // segment_length) * segment_length, positions - 47)
        lower = jnp.broadcast_to(lower[None], (LOCAL_BATCH, length))
        valid = jnp.ones((LOCAL_BATCH, length), dtype=jnp.bool_).at[:, 0].set(False)
        output, _ = segmented_flash_attention_forward(
            q,
            k,
            v,
            lower,
            valid,
            softmax_scale=128**-0.5,
            kernel_config=sm100_flash4_cute_kernel_config(),
            q_offset=jnp.zeros((1,), jnp.int32),
        )
        return AttentionCapture(q, k, v, output, dout, lower, valid, jnp.float32(0), jnp.float32(1))

    return jax.shard_map(local, out_specs=_capture_specs(), check_vma=False)()


def _array_metadata(capture: AttentionCapture) -> dict:
    return {
        name: {
            "global_shape": list(value.shape),
            "dtype": str(value.dtype),
            "weak_type": value.weak_type,
            "sharding": str(value.sharding),
            "addressable": [
                {"shape": list(shard.data.shape), "device": str(shard.device), "index": str(shard.index)}
                for shard in value.addressable_shards
            ],
        }
        for name, value in zip(AttentionCapture._fields, capture, strict=True)
    }


def _write_lowered(root: StoragePath, name: str, lowered) -> dict:
    # EqxLowered publicly exposes as_text(); do not unwrap private JIT functions.
    text = lowered.as_text()
    encoded = text.encode()
    if len(encoded) > MAX_HLO_BYTES:
        raise ValueError(f"Lowered IR exceeds the probe's {MAX_HLO_BYTES}-byte bound: {len(encoded)}")
    compressed = gzip.compress(encoded, mtime=0)
    artifact = root / f"{name}-lowered.mlir.gz"
    artifact.write_bytes(compressed)
    calls = []
    for line in text.splitlines():
        target = re.search(r"custom_call\s+@([^\s(]+)", line)
        if target is None:
            continue
        config_match = re.search(r'backend_config\s*=\s*("(?:[^"\\]|\\.)*")', line)
        calls.append(
            {
                "ordinal": len(calls),
                "target": target.group(1),
                "operation_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                "quoted_backend_config_sha256": (
                    hashlib.sha256(config_match.group(1).encode()).hexdigest() if config_match is not None else None
                ),
                "line_bytes": len(line.encode()),
            }
        )
    (root / f"{name}-custom-calls.json").write_text(json.dumps(calls, allow_nan=False))
    kernels = _cutlass_compile_metadata()
    (root / f"{name}-cutlass-cache.json").write_text(json.dumps(kernels, allow_nan=False))
    return {
        "artifact": str(artifact),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "uncompressed_bytes": len(encoded),
        "compressed_bytes": len(compressed),
        "custom_call_count": len(calls),
        "cutlass_kernel_count": len(kernels),
    }


def _cutlass_compile_metadata() -> list[dict]:
    # Pinned CuTe's exact API is also read by levanter.cutlass_kernel_cache.
    # Preserve hashes/identities only; the lowered IR already contains blobs.
    compile_module = importlib.import_module("cutlass.jax.compile")
    records = []
    for (launcher, spec), result in compile_module._CUTLASS_COMPILE_CACHE.copy().items():
        specification = repr(spec)
        records.append(
            {
                "launcher_identity": getattr(launcher, "_levanter_cute_kernel_identity", None),
                "source_covered": getattr(launcher, "_levanter_cute_kernel_source_covered", False),
                "specification": specification,
                "specification_sha256": hashlib.sha256(specification.encode()).hexdigest(),
                "module_sha256": hashlib.sha256(result.module).hexdigest(),
                "module_bytes": len(result.module),
                "fingerprint_hex": result.fingerprint.hex(),
            }
        )
    return sorted(records, key=lambda record: (str(record["launcher_identity"]), record["specification_sha256"]))


def run_compile_probe(request: dict) -> None:
    """Run the bounded distributed probe from an importable worker identity."""
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=True,
        moe_implementation=RAGGED_MOE_IMPLEMENTATION,
        remat_mode=OFFLOAD_CARRY_REMAT_MODE,
        processes_per_task=1,
    )
    faulthandler.dump_traceback_later(TRACEBACK_INTERVAL, repeat=True)
    DistributedConfig().initialize()
    install_gpu_compile_guard()
    configure_logging(logging.INFO)
    if jax.process_count() != 2 or jax.local_device_count() != 1 or jax.device_count() != 2:
        raise ValueError("Compile probe requires two distributed processes with one GPU each")
    install(cutlass_kernel_cache())
    root = StoragePath(request["output_root"]) / f"rank-{jax.process_index()}"
    metadata = {
        "source_revision": request["source_revision"],
        "evidence_kind": "synthetic_compile_probe_no_checkpoint_or_training",
        "compile_guard": "iris.jax.compile_guard installed; required numerical gates unchanged",
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "cutlass_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "flash_attention_version": importlib.metadata.version("flash-attn-4"),
        "probe_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "devices": [str(device) for device in jax.devices()],
        "local_device_kinds": [device.device_kind for device in jax.local_devices()],
        "local_gpu_compute_capability": gpu_compute_capability(),
        "kernel_cache_device_architecture": _device_architecture(),
        "local_batch": LOCAL_BATCH,
        "global_kv_duplication": 1,
        "layer_index": LAYER_INDEX,
        "query_count": QUERY_COUNT,
        "required_zero_gate_names": list(REQUIRED_ZERO_GATE_NAMES),
        "xla_flags": shlex.split(os.environ.get("XLA_FLAGS", "")),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "XLA_FLAGS",
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
                "JAX_COMPILATION_CACHE_DIR",
                "PYTHONHASHSEED",
                "CUTE_DSL_DEBUG",
                "CUTE_DSL_LINEINFO",
                "CUTE_DSL_LOC_TRACEBACKS",
            )
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata, allow_nan=False))
    phases = []
    started = time.monotonic()

    def phase(name: str, **details) -> None:
        phases.append(
            {"phase": name, "utc": datetime.now(UTC).isoformat(), "elapsed": time.monotonic() - started, **details}
        )
        (root / "phases.json").write_text(json.dumps(phases, allow_nan=False))
        handler.publish()
        logger.info("COMPILE_PROBE_PHASE rank=%d phase=%s", jax.process_index(), name)

    key_logger = logging.getLogger("jax._src.cache_key")
    previous_level, previous_propagation = key_logger.level, key_logger.propagate
    with tempfile.TemporaryDirectory(prefix="hero-compile-probe-") as directory:
        handler = CacheKeyFileHandler(Path(directory) / "cache-key-components.log", root / "cache-key-components.log.gz")
        key_logger.setLevel(logging.DEBUG)
        key_logger.propagate = False
        key_logger.addHandler(handler)
        try:
            results = []
            with jax.set_mesh(compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)):
                for length, heads in PROBE_CASES:
                    case = f"length-{length}-heads-{heads}"
                    phase(f"{case}-synthetic-lower-begin")
                    lowered_capture = synthetic_capture.lower(length=length, heads=heads)
                    phase(
                        f"{case}-synthetic-lower-complete",
                        lowered=_write_lowered(root, f"{case}-synthetic", lowered_capture),
                    )
                    phase(f"{case}-synthetic-compile-begin")
                    compiled_capture = lowered_capture.compile()
                    phase(f"{case}-synthetic-compile-complete")
                    capture = compiled_capture()
                    jax.block_until_ready(capture)
                    phase(f"{case}-synthetic-ready", arrays=_array_metadata(capture))
                    phase(f"{case}-replay-lower-begin")
                    lowered = replay_attention_mitigations.lower(
                        capture, layer_index=LAYER_INDEX, query_count=QUERY_COUNT
                    )
                    phase(f"{case}-replay-lower-complete", lowered=_write_lowered(root, f"{case}-replay", lowered))
                    phase(f"{case}-replay-compile-begin")
                    compiled = lowered.compile()
                    phase(f"{case}-replay-compile-complete")
                    replay = compiled(capture, layer_index=LAYER_INDEX, query_count=QUERY_COUNT)
                    phase(f"{case}-replay-dispatched")
                    jax.block_until_ready(replay)
                    host_gates = np.asarray(replay.gates.addressable_shards[0].data).reshape(-1)
                    if not np.isfinite(host_gates).all():
                        raise ValueError("Synthetic replay produced nonfinite gate evidence")
                    gates = dict(zip(REPLAY_GATE_NAMES, host_gates.tolist(), strict=True))
                    (root / f"{case}-gates.json").write_text(json.dumps(gates, allow_nan=False))
                    failed = {name: gates[name] for name in REQUIRED_ZERO_GATE_NAMES if gates[name] != 0}
                    if failed:
                        raise ValueError(f"Synthetic replay failed required numerical gates: {failed}")
                    results.append({"length": length, "query_heads": heads, "gates": gates})
                    phase(f"{case}-replay-ready-all-required-gates-pass")
                    jax.tree.map(lambda array: array.delete(), replay)
                    jax.tree.map(lambda array: array.delete(), capture)
            (root / "results.json").write_text(json.dumps({**metadata, "cases": results}, allow_nan=False))
            phase("complete")
        except Exception as exc:
            phase("failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            handler.publish()
            key_logger.removeHandler(handler)
            key_logger.setLevel(previous_level)
            key_logger.propagate = previous_propagation
            handler.close()
            faulthandler.cancel_dump_traceback_later()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, help="Explicit private object-storage output URI")
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    worker = importlib.import_module("experiments.grug.moe_hero_ep.attention_mitigation_compile_probe")
    dispatch_grug_training_run(
        run_id=args.run_id,
        config={"output_root": args.output_root, "source_revision": args.source_revision},
        local_entrypoint=worker.run_compile_probe,
        resources=ResourceConfig.with_gpu("GB200", count=1, cpu=24, ram="128g", disk="64g", replicas=2),
        processes_per_task=1,
        max_retries_failure=0,
        max_task_failures=0,
    )


if __name__ == "__main__":
    main()
