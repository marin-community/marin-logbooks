# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select actual native attention objects for a bounded frozen diagnostic.

The selected object follows CuTe's normal cache-hit contract: its real bytes and
SHA256 populate CompileResult. Every caller first agrees on source, specification
and runtime identity. The ordinary distributed XLA compile guard remains enabled.
This does not establish deterministic code generation or numerical equivalence
of independently generated objects. No object enters the production cache.
"""

import base64
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from functools import cache
from pathlib import Path

import jax
from levanter.cutlass_kernel_cache import _device_architecture, gpu_compute_capability
from levanter.utils.jax_utils import multihost_allgather_sync
from rigging.cache import directory_content_hash, file_content_hash, workspace_lock_hash
from rigging.filesystem.storage_path import StoragePath

MAX_MODULE_BYTES = 512 * 1024
MAX_ENVELOPE_BYTES = 1024 * 1024
MAX_IDENTITY_BYTES = 64 * 1024
MAX_PRODUCER_OBJECT_BYTES = 32 * 1024 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_ERROR_BYTES = 8192
MAX_CALLS = 4096
FAILURE_EVIDENCE_RESERVE = MAX_IDENTITY_BYTES + 2 * MAX_ERROR_BYTES + 4096


class NativeCompilation(StrEnum):
    INDEPENDENT = "independent"
    SHARED_OBJECT = "shared_object"


def _encoded(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _error_text(exc: Exception) -> str:
    return str(exc).encode()[:MAX_ERROR_BYTES].decode(errors="ignore")


def _participant_rows(rows: list[dict], process_ids: tuple[int, ...], phase: str) -> dict[int, dict]:
    indexed = {row["process_id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(process_ids):
        raise ValueError("Native object exchange returned different participants")
    if any(row["phase"] != phase for row in rows):
        raise ValueError("Native object exchange entered different phases")
    return indexed


def shared_native_module(
    identity: dict,
    *,
    process_id: int,
    process_ids: tuple[int, ...],
    exchange: Callable[[dict], list[dict]],
    compile_module: Callable[[], bytes],
) -> tuple[bytes, dict]:
    """Exchange an agreed request and actual producer object over coordinator RPCs."""
    encoded_identity = _encoded(identity)
    if len(encoded_identity) > MAX_IDENTITY_BYTES:
        raise ValueError(f"Native compile identity exceeds {MAX_IDENTITY_BYTES} bytes")
    identities = _participant_rows(
        exchange({"phase": "identity", "process_id": process_id, "identity": identity}), process_ids, "identity"
    )
    if any(row["identity"] != identity for row in identities.values()):
        raise ValueError(
            "Native compile call order, source, specification or runtime identity differs between processes"
        )
    identity_sha256 = hashlib.sha256(encoded_identity).hexdigest()
    producer = min(process_ids)
    producer_error = None
    payload = {"status": "waiting"}
    if process_id == producer:
        try:
            module = compile_module()
            if len(module) > MAX_MODULE_BYTES:
                raise ValueError(f"Native object has {len(module)} bytes, above the {MAX_MODULE_BYTES}-byte bound")
            payload = {
                "status": "complete",
                "module_base64": base64.b64encode(module).decode("ascii"),
                "module_sha256": hashlib.sha256(module).hexdigest(),
                "module_bytes": len(module),
            }
            if len(_encoded(payload)) > MAX_ENVELOPE_BYTES:
                raise ValueError(f"Native object envelope exceeds {MAX_ENVELOPE_BYTES} bytes")
        except Exception as exc:
            producer_error = exc
            payload = {"status": "failed", "error_type": type(exc).__name__, "error": _error_text(exc)}
    objects = _participant_rows(
        exchange(
            {
                "phase": "object",
                "process_id": process_id,
                "identity_sha256": identity_sha256,
                "object": payload,
            }
        ),
        process_ids,
        "object",
    )
    if any(row["identity_sha256"] != identity_sha256 for row in objects.values()):
        raise ValueError("Native object exchange changed its agreed request")
    if any(row["object"]["status"] != "waiting" for peer, row in objects.items() if peer != producer):
        raise ValueError("A non-producer supplied a native object")
    selected = objects[producer]["object"]
    if len(_encoded(selected)) > MAX_ENVELOPE_BYTES:
        raise ValueError(f"Received native object envelope exceeds {MAX_ENVELOPE_BYTES} bytes")
    if selected["status"] == "failed":
        raise RuntimeError(
            f"Native object producer failed: {selected['error_type']}: {selected['error']}"
        ) from producer_error
    if selected["status"] != "complete":
        raise ValueError("Native object producer did not provide a terminal result")
    module = base64.b64decode(selected["module_base64"], validate=True)
    if len(module) > MAX_MODULE_BYTES or len(module) != selected["module_bytes"]:
        raise ValueError("Received native object has an invalid byte count")
    digest = hashlib.sha256(module).hexdigest()
    if digest != selected["module_sha256"]:
        raise ValueError("Received native object SHA256 does not match its actual bytes")
    return module, {
        "identity_sha256": identity_sha256,
        "producer": producer,
        "module_sha256": digest,
        "module_bytes": len(module),
    }


@cache
def _source_identity(source_revision: str) -> dict:
    kernel_cache = importlib.import_module("levanter.cutlass_kernel_cache")
    assert kernel_cache.__file__ is not None
    kernel_cache_path = Path(kernel_cache.__file__).resolve()
    return {
        "source_revision": source_revision,
        "experiment_sources_sha256": directory_content_hash(Path(__file__).resolve().parent),
        "grug_sources_sha256": directory_content_hash(kernel_cache_path.parent / "grug"),
        "kernel_cache_source_sha256": file_content_hash(kernel_cache_path),
        "dependency_lock_sha256": workspace_lock_hash(Path(__file__)),
    }


def _runtime_identity() -> dict:
    return {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "nvidia-cutlass-dsl", "flash-attn-4", "cuda-bindings", "apache-tvm-ffi")
        },
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": list(sys.version_info[:3]),
        "python_cache_tag": sys.implementation.cache_tag,
        "device_architecture": _device_architecture(),
        "gpu_compute_capability": gpu_compute_capability(),
        "device_kinds": sorted({device.device_kind for device in jax.local_devices()}),
        # Dump destination paths are declared observation outputs. Other native
        # options remain part of the agreement; no HLO or fingerprint is edited.
        "native_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "CUTE_DSL_KEEP",
                "CUTE_DSL_DEBUG",
                "CUTE_DSL_LINEINFO",
                "CUTE_DSL_LOC_TRACEBACKS",
                "CUTE_DSL_COMPILER_OPT",
                "CUTE_DSL_ARCH",
                "CUTE_DSL_LIBS",
            )
        },
    }


@contextmanager
def native_compilation(
    mode: NativeCompilation, *, evidence_root: StoragePath, source_revision: str, timeout: float
) -> Iterator[None]:
    """Temporarily select real objects for an attention replay diagnostic.

    All native calls rendezvous, including local cache hits. The producer uses
    CuTe's original compile entry point, bypassing the persistent cache wrapper.
    Each process reconstructs the normal cache result from identical object bytes.
    """
    if mode is NativeCompilation.INDEPENDENT:
        yield
        return
    if mode is not NativeCompilation.SHARED_OBJECT:
        raise ValueError(f"Unknown native compilation mode: {mode}")
    primitive = importlib.import_module("cutlass.jax.primitive")
    compiler = importlib.import_module("cutlass.jax.compile")
    original = primitive.get_or_compile_kernel
    provenance = {"source": _source_identity(source_revision), "runtime": _runtime_identity()}
    provenance_encoded = _encoded(provenance)
    (evidence_root / "provenance.json").write_bytes(provenance_encoded)
    provenance_sha256 = hashlib.sha256(provenance_encoded).hexdigest()
    process_id = jax.process_index()
    process_ids = tuple(range(jax.process_count()))
    ordinal = 0
    evidence_bytes = len(provenance_encoded)
    producer_object_bytes = 0
    published_objects = set()
    mutex = threading.Lock()

    def exchange(value: dict) -> list[dict]:
        return multihost_allgather_sync(value, timeout=timeout, process_ids=process_ids)

    def get_or_compile_kernel(fn, spec):
        nonlocal ordinal, evidence_bytes, producer_object_bytes
        with mutex:
            if ordinal >= MAX_CALLS:
                raise ValueError(f"Native object context exceeded {MAX_CALLS} calls")
            identity = {
                "ordinal": ordinal,
                "launcher_identity": getattr(fn, "_levanter_cute_kernel_identity", None),
                "specification": repr(spec),
                "provenance_sha256": provenance_sha256,
            }
            if identity["launcher_identity"] is None or " at 0x" in identity["specification"]:
                raise ValueError("Native object selection requires a stable launcher and specification")
            call = ordinal
            ordinal += 1
            try:
                module, selected = shared_native_module(
                    identity,
                    process_id=process_id,
                    process_ids=process_ids,
                    exchange=exchange,
                    compile_module=lambda: compiler.get_or_compile_kernel(fn, spec).module,
                )
                record = {"identity": identity, "status": "complete", **selected}
                digest = selected["module_sha256"]
                if process_id == selected["producer"] and digest not in published_objects:
                    if producer_object_bytes + len(module) > MAX_PRODUCER_OBJECT_BYTES:
                        raise ValueError(f"Producer objects exceed the {MAX_PRODUCER_OBJECT_BYTES}-byte context bound")
                    (evidence_root / "objects" / f"{digest}.o").write_bytes(module)
                    producer_object_bytes += len(module)
                    published_objects.add(digest)
            except Exception as exc:
                record = {
                    "identity": identity,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": _error_text(exc),
                }
                (evidence_root / "failed.json").write_bytes(_encoded(record))
                raise
            encoded_record = _encoded(record)
            if evidence_bytes + len(encoded_record) > MAX_EVIDENCE_BYTES - FAILURE_EVIDENCE_RESERVE:
                raise ValueError(f"Native object evidence exceeds the {MAX_EVIDENCE_BYTES}-byte context bound")
            (evidence_root / "calls" / f"call-{call:04d}.json").write_bytes(encoded_record)
            evidence_bytes += len(encoded_record)
            result = compiler.CompileResult(module=module, fingerprint=hashlib.sha256(module).digest(), spec=spec)
            compiler._CUTLASS_COMPILE_CACHE[(fn, spec)] = result
            return result

    primitive.get_or_compile_kernel = get_or_compile_kernel
    try:
        yield
    finally:
        primitive.get_or_compile_kernel = original
