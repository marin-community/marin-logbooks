# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode native SM100 accumulators without rounding their FP32 values.

The native buffer is a packed MMA accumulator, not flattened logical BSHD.
This experiment preserves the pinned postprocessor's BF16 MMA construction and
S2R mapping, then exports its FP32 register values through its own TMEM identity
coordinates before the BF16 conversion. Installed third-party files are never
modified. Their original copyright header is retained in the isolated copy.
"""

import hashlib
import importlib.util
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug.attention._fa4_cute_backend import (
    _import_cutlass_cute,
    _native_backward_postprocess_specs,
)
from levanter.grug.attention._fa4_cute_kernels import _import_cute_dependencies, _native_gmem_tensor_view

SM100_POSTPROCESS_SOURCE_SHA256 = "df88b299cb938197abc145c3a85262515d1d810633229d0f929239e8291e8ce8"
_SOURCE_MODULE = "flash_attn.cute.flash_bwd_postprocess"
_TYPE_ANCHOR = "        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):\n"
_CONVERSION_ANCHOR = "                # Convert tdQrdQaccum from fp32 to fp16/bf16\n"
_FINAL_COPY_ANCHOR = "            # Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem\n"
_FP32_EXPORT = """                if const_expr(mdQ.element_type == Float32):
                    assert self.arch == 100 and not self.use_2cta_instrs
                    assert self.tile_hdim == 128 and self.num_threads == 128
                    hero_identity = thr_copy_t2r.partition_D(tdQcdQ)
                    for hero_element in cutlass.range_constexpr(cute.size(acc)):
                        hero_coord = hero_identity[hero_element]
                        if hero_coord[0] < seqlen_q - m_block * self.tile_m and hero_coord[1] < head_dim:
                            gdQ[hero_coord] = acc[hero_element] * scale
                else:
"""


def _patched_postprocess_source(source: bytes) -> str:
    actual_digest = hashlib.sha256(source).hexdigest()
    if actual_digest != SM100_POSTPROCESS_SOURCE_SHA256:
        raise RuntimeError(
            f"Native postprocess source mismatch: expected {SM100_POSTPROCESS_SOURCE_SHA256}, got {actual_digest}"
        )
    text = source.decode()
    for anchor in (_TYPE_ANCHOR, _CONVERSION_ANCHOR, _FINAL_COPY_ANCHOR):
        if text.count(anchor) != 1:
            raise RuntimeError("Pinned postprocess export anchor is not unique")
    text = text.replace(
        _TYPE_ANCHOR,
        "        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16, Float32]):\n",
    )
    before_conversion, remainder = text.split(_CONVERSION_ANCHOR)
    conversion, final_copy = remainder.split(_FINAL_COPY_ANCHOR)
    conversion = _CONVERSION_ANCHOR + conversion
    final_copy = _FINAL_COPY_ANCHOR + final_copy
    # Keep every original cast/copy operation for BF16 outputs. FP32 exports
    # bypass them while retaining the original register decoding above.
    return (
        before_conversion
        + _FP32_EXPORT
        + "".join("    " + line if line.strip() else line for line in conversion.splitlines(keepends=True))
        + "            if const_expr(mdQ.element_type != Float32):\n"
        + "".join("    " + line if line.strip() else line for line in final_copy.splitlines(keepends=True))
    )


@lru_cache(maxsize=1)
def native_fp32_postprocess_module() -> ModuleType:
    """Load the hash-checked private postprocessor with FP32 register export."""
    source_spec = importlib.util.find_spec(_SOURCE_MODULE)
    if source_spec is None or source_spec.origin is None:
        raise ModuleNotFoundError(f"Cannot locate pinned native source: {_SOURCE_MODULE}")
    patched = _patched_postprocess_source(Path(source_spec.origin).read_bytes())
    digest = hashlib.sha256(patched.encode()).hexdigest()
    module_name = f"flash_attn.cute._hero_fp32_postprocess_{digest}"
    directory = Path(tempfile.mkdtemp(prefix="hero-fa4-accumulator-"))
    source_path = directory / f"{module_name.rsplit('.', 1)[-1]}.py"
    source_path.write_text(patched)
    module_spec = importlib.util.spec_from_file_location(module_name, source_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load isolated native source: {source_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


@cute_launcher_factory
def _native_accumulator_decode_launcher(modules: Any, *, tile_m: int) -> Any:
    deps = _import_cute_dependencies(modules)
    cutlass, cute, cuda = deps.cutlass, deps.cute, deps.cuda
    native = native_fp32_postprocess_module()
    # BF16 is intentional: it preserves the original MMA and S2R layout.
    postprocess = native.FlashAttentionBackwardPostprocess(
        cutlass.BFloat16,
        128,
        100,
        tile_m,
        num_threads=128,
        AtomLayoutMdQ=1,
        cluster_size=1,
        use_2cta_instrs=False,
    )
    as_gmem_tensor = _native_gmem_tensor_view(modules)

    @cute.jit
    def launch(stream: cuda.CUstream, accum: cute.Tensor, output: cute.Tensor):
        postprocess(as_gmem_tensor(accum), output, cutlass.Float32(1), None, None, None, None, stream)

    return launch


def decode_native_accumulator(accum: jax.Array, *, tensor_shape: tuple[int, int, int, int], tile_m: int) -> jax.Array:
    """Decode local one-CTA SM100 D128 accumulators to unscaled FP32 BSHD.

    Call inside the caller's shard_map. The required GPU acceptance check is
    bitwise agreement of ``decode(...).astype(bfloat16)`` with the untouched
    native postprocessor at scale one on nontrivial accumulator data.
    """
    batch, length, heads, width = tensor_shape
    rounded_length = ((length + tile_m - 1) // tile_m) * tile_m
    if width != 128 or tile_m != 128 or accum.dtype != jnp.float32:
        raise ValueError("Accumulator decoder requires SM100 128x128 BF16-MMA FP32 accumulators")
    if accum.shape != (batch, heads, rounded_length * width):
        raise ValueError("Accumulator shape does not match native padded BHS*D layout")
    modules = _import_cutlass_cute()
    inputs, outputs = _native_backward_postprocess_specs(modules, vector_elems=8)
    decode = cutlass_call(
        _native_accumulator_decode_launcher(modules, tile_m=tile_m),
        output_shape_dtype=(jax.ShapeDtypeStruct(tensor_shape, jnp.float32),),
        input_spec=inputs,
        output_spec=outputs,
        use_static_tensors=True,
    )
    return decode(accum)[0]
