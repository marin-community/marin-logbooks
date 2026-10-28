# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic distributed compile-key probe for the unchanged Hero mitigation replay.

This restores no checkpoint and performs no optimizer updates. Inputs are random
BF16 synthetic tensors, with four local sequences per GPU, packed causal/window
masks, all query heads, and one logical KV copy. The production replay and its
eleven required numerical gates are unchanged. All evidence goes only to the
explicit private output root, including potentially sensitive compiler IR.
"""

import argparse
import ast
import faulthandler
import gzip
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import logging
import os
import re
import shlex
import socket
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial, wraps
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from finestore.cache import PersistentKvCache
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
    MitigationReplay,
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
MAX_NATIVE_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_NATIVE_UPLOAD_BYTES = 64 * 1024 * 1024
TRACEBACK_INTERVAL = 120
PROBE_CASES = ((256, 8), (16384, 48))
NATIVE_INPUT_CAPTURE_SELECTION = {
    "launcher_factory": "levanter.grug.attention._fa4_cute_kernels.segmented_flash_attention_forward_sm100_launcher",
    "configuration": "Flash4CuteSm100ForwardConfig(tile=(128, 128), q_stage=1)",
    "query_sequence_length": 16384,
    "first_output_dtype": "float32",
}


@dataclass
class NativeCompilerArtifacts:
    directory: Path
    dump_directory: Path
    published_sha256: dict[str, str] = field(default_factory=dict)
    uploaded_bytes: int = 0
    kernel_records: list[dict] = field(default_factory=list)
    compiler_call_index: int = 0
    native_input_capture: dict | None = None


@contextmanager
def _native_compiler_artifact_directory(mode: str) -> Iterator[NativeCompilerArtifacts | None]:
    if mode == "none":
        yield None
        return
    if mode != "ir_ptx_cubin":
        raise ValueError(f"Unknown native compiler artifact mode: {mode}")
    previous = {name: os.environ.get(name) for name in ("CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR")}
    with tempfile.TemporaryDirectory(prefix="hero-native-compiler-") as directory:
        dump_directory = Path(directory) / "live"
        dump_directory.mkdir()
        os.environ["CUTE_DSL_KEEP"] = "ir,ptx,cubin"
        os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_directory)
        try:
            yield NativeCompilerArtifacts(Path(directory), dump_directory)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _native_dump_file_metadata(directory: Path) -> dict[str, tuple[int, int]]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Native compiler artifact must not be a symlink: {path}")
        if path.is_file():
            stat = path.stat()
            files[path.relative_to(directory).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return files


def _native_input_capture(fn, spec) -> dict | None:
    identity = getattr(fn, "_levanter_cute_kernel_identity", None)
    if identity is None:
        return None
    parts = identity.split("|")
    if (
        parts[0] != NATIVE_INPUT_CAPTURE_SELECTION["launcher_factory"]
        or f"config={NATIVE_INPUT_CAPTURE_SELECTION['configuration']}" not in parts
        or spec.in_args[0].shape[1] != NATIVE_INPUT_CAPTURE_SELECTION["query_sequence_length"]
        or np.dtype(spec.out_args[0].dtype) != np.dtype(NATIVE_INPUT_CAPTURE_SELECTION["first_output_dtype"])
    ):
        return None
    return {
        "selection": NATIVE_INPUT_CAPTURE_SELECTION,
        "launcher_identity": identity,
        "specification_sha256": hashlib.sha256(repr(spec).encode()).hexdigest(),
    }


def _snapshot_native_kernel(
    artifacts: NativeCompilerArtifacts, fn, spec, result, before: dict[str, tuple[int, int]], *, cache_hit: bool
) -> None:
    specification = repr(spec)
    module_sha256 = hashlib.sha256(result.module).hexdigest()
    record = {
        "ordinal": len(artifacts.kernel_records),
        "launcher_identity": getattr(fn, "_levanter_cute_kernel_identity", None),
        "specification": specification,
        "specification_sha256": hashlib.sha256(specification.encode()).hexdigest(),
        "module_sha256": module_sha256,
        "module_bytes": len(result.module),
        "exact_input_capture_selected": _native_input_capture(fn, spec) is not None,
    }
    if cache_hit:
        record.update(status="in_process_cache_hit", files=[])
    else:
        after = _native_dump_file_metadata(artifacts.dump_directory)
        changed = [name for name, metadata in after.items() if before.get(name) != metadata]
        record["changed_dump_files"] = changed
        if not changed:
            record.update(status="no_new_compiler_artifacts", files=[])
        else:
            snapshot = artifacts.directory / "kernels" / module_sha256 / str(record["ordinal"])
            snapshot.mkdir(parents=True)
            files = []
            for name in changed:
                source = artifacts.dump_directory / name
                data = source.read_bytes()
                destination = snapshot / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                files.append(
                    {
                        "path": destination.relative_to(artifacts.directory).as_posix(),
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                # These observation files belong to this wrapper; keep their
                # immutable copies without accumulating duplicate live copies.
                if source.name.startswith("compiler-call-"):
                    source.unlink()
            record.update(status="captured", files=files)
    artifacts.kernel_records.append(record)
    (artifacts.directory / "native-kernel-records.json").write_text(
        json.dumps(artifacts.kernel_records, allow_nan=False)
    )


@contextmanager
def _native_kernel_snapshots(artifacts: NativeCompilerArtifacts | None) -> Iterator[None]:
    if artifacts is None:
        yield
        return
    primitive = importlib.import_module("cutlass.jax.primitive")
    compile_module = importlib.import_module("cutlass.jax.compile")
    original = primitive.get_or_compile_kernel

    @wraps(original)
    def get_or_compile_kernel(fn, spec):
        cached = compile_module._CUTLASS_COMPILE_CACHE.get((fn, spec))
        before = _native_dump_file_metadata(artifacts.dump_directory) if cached is None else {}
        previous = artifacts.native_input_capture
        artifacts.native_input_capture = _native_input_capture(fn, spec)
        try:
            result = original(fn, spec)
        finally:
            artifacts.native_input_capture = previous
        _snapshot_native_kernel(artifacts, fn, spec, result, before, cache_hit=cached is result)
        return result

    primitive.get_or_compile_kernel = get_or_compile_kernel
    try:
        yield
    finally:
        primitive.get_or_compile_kernel = original


@contextmanager
def _native_compiler_calls(artifacts: NativeCompilerArtifacts | None) -> Iterator[None]:
    if artifacts is None:
        yield
        return
    compiler = importlib.import_module("cutlass.base_dsl.compiler").Compiler
    original = compiler.compile
    signature = inspect.signature(original)

    @wraps(original)
    def compile_native(self, module, pipeline, *args, **kwargs):
        bound = signature.bind(self, module, pipeline, *args, **kwargs)
        bound.apply_defaults()
        arguments = {name: value for name, value in bound.arguments.items() if name not in ("self", "module")}
        prefix = artifacts.dump_directory / f"compiler-call-{artifacts.compiler_call_index:04d}"
        artifacts.compiler_call_index += 1
        record = {"arguments": arguments, "exact_input_capture": artifacts.native_input_capture}
        if artifacts.native_input_capture is not None:
            bytecode = io.BytesIO()
            module.operation.write_bytecode(bytecode)
            serialized = bytecode.getvalue()
            assembly = module.operation.get_asm(enable_debug_info=True).encode()
            prefix.with_suffix(".mlirbc").write_bytes(serialized)
            prefix.with_suffix(".mlir").write_bytes(assembly)
            record.update(
                module_bytecode_sha256=hashlib.sha256(serialized).hexdigest(),
                module_bytecode_bytes=len(serialized),
                module_asm_sha256=hashlib.sha256(assembly).hexdigest(),
                module_asm_bytes=len(assembly),
            )
        prefix.with_suffix(".json").write_text(json.dumps(record, allow_nan=False))
        return original(self, module, pipeline, *args, **kwargs)

    compiler.compile = compile_native
    try:
        yield
    finally:
        compiler.compile = original


def _publish_native_compiler_artifacts(artifacts: NativeCompilerArtifacts, root: StoragePath) -> dict:
    files = []
    total_bytes = 0
    for path in sorted(artifacts.directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Native compiler artifact must not be a symlink: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append((path, size))
    manifest = {
        "files": [],
        "total_bytes": total_bytes,
        "process_byte_bound": MAX_NATIVE_ARTIFACT_BYTES,
        "compressed_upload_byte_bound": MAX_NATIVE_UPLOAD_BYTES,
        "uploaded_bytes": artifacts.uploaded_bytes,
    }
    if total_bytes > MAX_NATIVE_ARTIFACT_BYTES:
        reason = f"Native compiler artifacts exceed the {MAX_NATIVE_ARTIFACT_BYTES}-byte raw process bound"
        manifest.update(status="bounds_exceeded", reason=reason)
        manifest["files"] = [
            {"path": path.relative_to(artifacts.directory).as_posix(), "bytes": size} for path, size in files
        ]
        (root / "native-compiler-manifest.json").write_text(json.dumps(manifest, allow_nan=False))
        raise ValueError(reason)
    records = []
    uploads = []
    pending_upload_bytes = 0
    for path, size in files:
        relative = path.relative_to(artifacts.directory).as_posix()
        data = path.read_bytes()
        if len(data) != size:
            raise ValueError(f"Native compiler artifact changed while publishing: {path}")
        compressed = gzip.compress(data, mtime=0)
        digest = hashlib.sha256(data).hexdigest()
        records.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": digest,
                "compressed_bytes": len(compressed),
            }
        )
        if artifacts.published_sha256.get(relative) != digest:
            pending_upload_bytes += len(compressed)
            uploads.append((relative, digest, compressed))
    manifest["files"] = records
    if artifacts.uploaded_bytes + pending_upload_bytes > MAX_NATIVE_UPLOAD_BYTES:
        reason = f"Native compiler artifacts exceed the {MAX_NATIVE_UPLOAD_BYTES}-byte compressed process upload bound"
        manifest.update(status="bounds_exceeded", reason=reason, pending_upload_bytes=pending_upload_bytes)
        (root / "native-compiler-manifest.json").write_text(json.dumps(manifest, allow_nan=False))
        raise ValueError(reason)
    for relative, digest, compressed in uploads:
        (root / "native-compiler" / f"{relative}.gz").write_bytes(compressed)
        artifacts.published_sha256[relative] = digest
        artifacts.uploaded_bytes += len(compressed)
    manifest.update(status="complete", uploaded_bytes=artifacts.uploaded_bytes)
    (root / "native-compiler-manifest.json").write_text(json.dumps(manifest, allow_nan=False))
    return {"file_count": len(records), "total_bytes": total_bytes}


def _write_native_compiler_source_evidence(artifacts: NativeCompilerArtifacts) -> None:
    records = []
    for name in ("env_manager", "dsl", "compiler"):
        module = importlib.import_module(f"cutlass.base_dsl.{name}")
        source_path = Path(module.__file__)
        source = source_path.read_bytes()
        record = {"module": module.__name__, "sha256": hashlib.sha256(source).hexdigest()}
        if name == "dsl":
            text = source.decode()
            record["merge_gpu_link_libraries_source"] = [
                ast.get_source_segment(text, node)
                for node in ast.walk(ast.parse(text))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_merge_gpu_link_libraries"
            ]
        records.append(record)
    (artifacts.directory / "compiler-source-evidence.json").write_text(json.dumps(records, allow_nan=False))


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


@contextmanager
def _cache_key_evidence(root: StoragePath) -> Iterator[CacheKeyFileHandler]:
    key_logger = logging.getLogger("jax._src.cache_key")
    previous_level, previous_propagation = key_logger.level, key_logger.propagate
    with tempfile.TemporaryDirectory(prefix="hero-compile-probe-") as directory:
        handler = CacheKeyFileHandler(Path(directory) / "cache-key-components.log", root / "cache-key-components.log.gz")
        key_logger.setLevel(logging.DEBUG)
        key_logger.propagate = False
        key_logger.addHandler(handler)
        try:
            yield handler
        finally:
            handler.publish()
            key_logger.removeHandler(handler)
            key_logger.setLevel(previous_level)
            key_logger.propagate = previous_propagation
            handler.close()


def _phase_recorder(root: StoragePath, handler: CacheKeyFileHandler) -> Callable[..., None]:
    phases = []
    started = time.monotonic()

    def phase(name: str, **details) -> None:
        phases.append(
            {"phase": name, "utc": datetime.now(UTC).isoformat(), "elapsed": time.monotonic() - started, **details}
        )
        (root / "phases.json").write_text(json.dumps(phases, allow_nan=False))
        handler.publish()
        logger.info("COMPILE_EVIDENCE_PHASE rank=%d phase=%s", jax.process_index(), name)

    return phase


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


def _decode_mlir_bytes(value: str) -> bytes:
    """Decode MLIR byte strings, rejecting malformed or unknown escapes."""
    result = bytearray()
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\":
            if ord(char) > 127:
                raise ValueError("MLIR binary string contains an unescaped non-ASCII byte")
            result.append(ord(char))
            index += 1
            continue
        hexadecimal = value[index + 1 : index + 3]
        if len(hexadecimal) == 2 and all(digit in "0123456789abcdefABCDEF" for digit in hexadecimal):
            result.append(int(hexadecimal, 16))
            index += 3
        elif index + 1 < len(value) and value[index + 1] in ("\\", '"'):
            result.append(ord(value[index + 1]))
            index += 2
        else:
            raise ValueError(f"Unsupported MLIR binary-string escape at offset {index}")
    return bytes(result)


def _custom_call_metadata(text: str) -> list[dict]:
    calls = []
    for line in text.splitlines():
        target = re.search(r"custom_call\s+@([^\s(]+)", line)
        if target is None:
            continue
        record = {
            "ordinal": len(calls),
            "target": target.group(1),
            "operation_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
            "line_bytes": len(line.encode()),
        }
        config_start = line.find("mhlo.backend_config = {")
        if config_start >= 0:
            config = line[config_start:]
            key_match = re.search(r'\bkey\s*=\s*"((?:[^"\\]|\\.)*)"', config)
            module_match = re.search(r'\bmodule\s*=\s*"((?:[^"\\]|\\.)*)"', config)
            if key_match is not None and module_match is not None:
                key = _decode_mlir_bytes(key_match.group(1))
                module = _decode_mlir_bytes(module_match.group(1))
                module_sha256 = hashlib.sha256(module).hexdigest()
                record.update(
                    backend_config_module_sha256=module_sha256,
                    backend_config_module_bytes=len(module),
                    backend_config_key_hex=key.hex(),
                    module_sha256_equals_key=module_sha256 == key.hex(),
                )
        if target.group(1).startswith("CuteDSLRT_") and "backend_config_module_sha256" not in record:
            raise ValueError("CuTe custom call is missing its expected module/key byte attributes")
        calls.append(record)
    return calls


def _write_lowered(
    root: StoragePath, name: str, lowered, native_directory: NativeCompilerArtifacts | None = None
) -> dict:
    # EqxLowered publicly exposes as_text(); do not unwrap private JIT functions.
    text = lowered.as_text()
    encoded = text.encode()
    if len(encoded) > MAX_HLO_BYTES:
        raise ValueError(f"Lowered IR exceeds the probe's {MAX_HLO_BYTES}-byte bound: {len(encoded)}")
    compressed = gzip.compress(encoded, mtime=0)
    artifact = root / f"{name}-lowered.mlir.gz"
    artifact.write_bytes(compressed)
    calls = _custom_call_metadata(text)
    (root / f"{name}-custom-calls.json").write_text(json.dumps(calls, allow_nan=False))
    kernels = _cutlass_compile_metadata()
    (root / f"{name}-cutlass-cache.json").write_text(json.dumps(kernels, allow_nan=False))
    native_artifacts = (
        _publish_native_compiler_artifacts(native_directory, root) if native_directory is not None else None
    )
    return {
        "artifact": str(artifact),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "uncompressed_bytes": len(encoded),
        "compressed_bytes": len(compressed),
        "custom_call_count": len(calls),
        "cutlass_kernel_count": len(kernels),
        "native_compiler_artifacts": native_artifacts,
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


def _runtime_metadata() -> dict:
    return {
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "cutlass_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "flash_attention_version": importlib.metadata.version("flash-attn-4"),
        "probe_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hostname": socket.gethostname(),
        "placement": {
            key: os.environ.get(key)
            for key in (
                "IRIS_TASK_ID",
                "IRIS_NUM_TASKS",
                "IRIS_MULTIGPU_PROCESS_INDEX",
                "IRIS_MULTIGPU_PROCESS_COUNT",
                "IRIS_MULTIGPU_LOCAL_DEVICE_IDS",
            )
        },
        "devices": [str(device) for device in jax.devices()],
        "local_device_kinds": [device.device_kind for device in jax.local_devices()],
        "local_gpu_compute_capability": gpu_compute_capability(),
        "kernel_cache_device_architecture": _device_architecture(),
        "xla_flags": shlex.split(os.environ.get("XLA_FLAGS", "")),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
                "JAX_COMPILATION_CACHE_DIR",
                "PYTHONHASHSEED",
                "CUTE_DSL_DEBUG",
                "CUTE_DSL_LINEINFO",
                "CUTE_DSL_LOC_TRACEBACKS",
                "CUTE_DSL_KEEP",
                "CUTE_DSL_DUMP_DIR",
                "CUTE_DSL_COMPILER_OPT",
                "CUTE_DSL_ARCH",
                "CUTE_DSL_LIBS",
            )
        },
    }


def _compile_replay(
    capture: AttentionCapture,
    *,
    root: StoragePath,
    name: str,
    layer_index: int,
    query_count: int,
    phase: Callable[..., None],
    native_directory: NativeCompilerArtifacts | None = None,
) -> MitigationReplay:
    phase(f"{name}-lower-begin", arrays=_array_metadata(capture))
    lowered = replay_attention_mitigations.lower(capture, layer_index=layer_index, query_count=query_count)
    phase(f"{name}-lower-complete", lowered=_write_lowered(root, name, lowered, native_directory))
    phase(f"{name}-compile-begin")
    compiled = lowered.compile()
    phase(f"{name}-compile-complete")
    replay = compiled(capture, layer_index=layer_index, query_count=query_count)
    phase(f"{name}-dispatched")
    return replay


def replay_with_compile_evidence(
    capture: AttentionCapture, *, root: StoragePath, layer_index: int, query_count: int
) -> MitigationReplay:
    """Run the unchanged replay while preserving bounded private compile evidence.

    ``root`` must identify this process's private evidence directory. Inputs stay
    dynamic, including actual checkpoint-derived values; only shapes, placement,
    compiler IR and kernel identities are recorded. The guard remains installed
    and numerical acceptance remains with the caller's unchanged gate checks.
    """
    install_gpu_compile_guard()
    metadata = {
        **_runtime_metadata(),
        "layer_index": layer_index,
        "query_count": query_count,
        "evidence_kind": "compile_evidence_for_supplied_capture",
        "required_zero_gate_names": list(REQUIRED_ZERO_GATE_NAMES),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, allow_nan=False))
    with _cache_key_evidence(root) as handler:
        phase = _phase_recorder(root, handler)
        try:
            return _compile_replay(
                capture, root=root, name="replay", layer_index=layer_index, query_count=query_count, phase=phase
            )
        except Exception as exc:
            phase("failed", error_type=type(exc).__name__, error=str(exc))
            raise


def run_compile_probe(request: dict) -> None:
    """Run the bounded distributed probe from an importable worker identity."""
    with _native_compiler_artifact_directory(request["native_artifacts"]) as native_directory:
        _run_compile_probe(request, native_directory=native_directory)


def _run_compile_probe(request: dict, *, native_directory: NativeCompilerArtifacts | None) -> None:
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=True,
        moe_implementation=RAGGED_MOE_IMPLEMENTATION,
        remat_mode=OFFLOAD_CARRY_REMAT_MODE,
        processes_per_task=request["gpus_per_task"],
    )
    faulthandler.dump_traceback_later(TRACEBACK_INTERVAL, repeat=True)
    DistributedConfig().initialize()
    install_gpu_compile_guard()
    configure_logging(logging.INFO)
    world = request["gpus_per_task"] * request["replicas"]
    if jax.process_count() != world or jax.local_device_count() != 1 or jax.device_count() != world:
        raise ValueError("Compile probe requires one distributed process per requested GPU")
    if request["kernel_cache"] == "persistent":
        cache = cutlass_kernel_cache()
    elif request["kernel_cache"] == "in_memory":
        cache = PersistentKvCache.in_memory()
    else:
        raise ValueError(f"Unknown declared kernel cache mode: {request['kernel_cache']}")
    install(cache)
    if native_directory is not None:
        _write_native_compiler_source_evidence(native_directory)
    root = StoragePath(request["output_root"]) / f"rank-{jax.process_index()}"
    metadata = {
        **_runtime_metadata(),
        "source_revision": request["source_revision"],
        "evidence_kind": "synthetic_compile_probe_no_checkpoint_or_training",
        "compile_guard": "iris.jax.compile_guard installed; required numerical gates unchanged",
        "local_batch": LOCAL_BATCH,
        "global_kv_duplication": 1,
        "layer_index": LAYER_INDEX,
        "query_count": QUERY_COUNT,
        "required_zero_gate_names": list(REQUIRED_ZERO_GATE_NAMES),
        "kernel_cache": request["kernel_cache"],
        "gpus_per_task": request["gpus_per_task"],
        "replicas": request["replicas"],
        "native_artifacts": request["native_artifacts"],
        "native_input_capture_selection": NATIVE_INPUT_CAPTURE_SELECTION if native_directory is not None else None,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, allow_nan=False))
    with (
        _cache_key_evidence(root) as handler,
        _native_kernel_snapshots(native_directory),
        _native_compiler_calls(native_directory),
    ):
        phase = _phase_recorder(root, handler)
        try:
            results = []
            with jax.set_mesh(compact_grug_mesh(expert_axis_size=world, replica_axis_size=1)):
                for length, heads in PROBE_CASES:
                    case = f"length-{length}-heads-{heads}"
                    phase(f"{case}-synthetic-lower-begin")
                    lowered_capture = synthetic_capture.lower(length=length, heads=heads)
                    phase(
                        f"{case}-synthetic-lower-complete",
                        lowered=_write_lowered(root, f"{case}-synthetic", lowered_capture, native_directory),
                    )
                    phase(f"{case}-synthetic-compile-begin")
                    compiled_capture = lowered_capture.compile()
                    phase(f"{case}-synthetic-compile-complete")
                    capture = compiled_capture()
                    jax.block_until_ready(capture)
                    phase(f"{case}-synthetic-ready", arrays=_array_metadata(capture))
                    replay = _compile_replay(
                        capture,
                        root=root,
                        name=f"{case}-replay",
                        layer_index=LAYER_INDEX,
                        query_count=QUERY_COUNT,
                        phase=phase,
                        native_directory=native_directory,
                    )
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
            if native_directory is not None:
                _publish_native_compiler_artifacts(native_directory, root)
            faulthandler.cancel_dump_traceback_later()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, help="Explicit private object-storage output URI")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--kernel-cache", required=True, choices=("persistent", "in_memory"))
    parser.add_argument("--gpus-per-task", required=True, type=int, choices=(1, 2, 4))
    parser.add_argument("--replicas", required=True, type=int)
    parser.add_argument("--native-artifacts", required=True, choices=("none", "ir_ptx_cubin"))
    args = parser.parse_args()
    if args.replicas <= 0:
        parser.error("--replicas must be positive")
    worker = importlib.import_module("experiments.grug.moe_hero_ep.attention_mitigation_compile_probe")
    dispatch_grug_training_run(
        run_id=args.run_id,
        config={
            "output_root": args.output_root,
            "source_revision": args.source_revision,
            "kernel_cache": args.kernel_cache,
            "gpus_per_task": args.gpus_per_task,
            "replicas": args.replicas,
            "native_artifacts": args.native_artifacts,
        },
        local_entrypoint=worker.run_compile_probe,
        resources=ResourceConfig.with_gpu(
            "GB200",
            count=args.gpus_per_task,
            cpu=24 * args.gpus_per_task,
            ram=f"{128 * args.gpus_per_task}g",
            disk="64g",
            replicas=args.replicas,
        ),
        processes_per_task=args.gpus_per_task,
        max_retries_failure=0,
        max_task_failures=0,
    )


if __name__ == "__main__":
    main()
