# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated, source-pinned SM100 backward instrumentation for operand panels."""

import hashlib
import importlib.util
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType

SM100_BACKWARD_SOURCE_SHA256 = "448b189cf4a12a67cb1e3efbb7e5eb38b37a2fc921a1835454b6ac679a6bf408"
_SOURCE_MODULE = "flash_attn.cute.flash_bwd_sm100"
_PROBABILITY_ANCHOR = "                    utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])\n"
_SCORE_GRADIENT_ANCHOR = "                    utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)\n"
_CAPTURE_FRAGMENT = """
                    # Capture exactly the BF16 operands consumed by the native MMAs.
                    # Aux tensors 0:3 keep the production packed-mask contract.
                    assert not self.use_2cta_instrs
                    assert not self.pack_gqa
                    hero_positions = aux_data.tensors[3]
                    hero_panel = aux_data.tensors[{panel_slot}]
                    hero_coords = tScS_t2r[None, stage, 0, 0]
                    hero_operand = {operand}
                    for hero_element in cutlass.range_constexpr(cute.size(hero_operand)):
                        hero_key = n_block * self.tile_n + hero_coords[hero_element][0]
                        hero_query = m_block * self.tile_m + hero_coords[hero_element][1]
                        if hero_key < seqlen.seqlen_k and hero_query < seqlen.seqlen_q:
                            for hero_selected in cutlass.range_constexpr(hero_positions.shape[1]):
                                if hero_query == hero_positions[batch_idx, hero_selected]:
                                    hero_index = (batch_idx, hero_selected, head_idx, hero_key)
                                    hero_panel[hero_index] = hero_operand[hero_element]
"""


@lru_cache(maxsize=1)
def native_sm100_operand_capture_module() -> ModuleType:
    """Load a private native backward class that exports actual post-cast panels.

    The caller supplies ``AuxData.tensors`` as ``(lower_bounds, valid, q_offset,
    selected_positions, probability_operand, score_gradient_operand)``. Positions
    have shape ``[batch, selected_queries]`` and both BF16 panel outputs have
    shape ``[batch, selected_queries, query_heads, keys]``. Zero both panels
    before launch: packed-mask sparsity can skip whole key tiles.

    Only the production one-CTA, unpacked-GQA SM100 configuration is supported.
    This preserves native arithmetic and gradient accumulators, but adds global
    writes and is an instrumented diagnostic rather than a fused GProj kernel.
    Installed source is never changed; CuTe compiles a separate module copied
    into process-local temporary storage after its source hash is checked.
    """
    source_spec = importlib.util.find_spec(_SOURCE_MODULE)
    if source_spec is None or source_spec.origin is None:
        raise ModuleNotFoundError(f"Cannot locate the pinned native source: {_SOURCE_MODULE}")
    source = Path(source_spec.origin).read_bytes()
    actual_digest = hashlib.sha256(source).hexdigest()
    if actual_digest != SM100_BACKWARD_SOURCE_SHA256:
        raise RuntimeError(
            f"Native SM100 backward source mismatch: expected {SM100_BACKWARD_SOURCE_SHA256}, got {actual_digest}"
        )
    text = source.decode()
    patched = text
    for anchor, operand, panel_slot in (
        (_PROBABILITY_ANCHOR, "tSrP_r2t[None, stage, 0, 0]", 4),
        (_SCORE_GRADIENT_ANCHOR, "tdPrdS_cvt", 5),
    ):
        if patched.count(anchor) != 1:
            raise RuntimeError("Pinned SM100 backward operand-capture anchor is not unique")
        fragment = _CAPTURE_FRAGMENT.format(operand=operand, panel_slot=panel_slot)
        patched = patched.replace(anchor, anchor + fragment)
    patched_digest = hashlib.sha256(patched.encode()).hexdigest()
    module_name = f"flash_attn.cute._hero_operand_capture_{patched_digest}"
    directory = Path(tempfile.mkdtemp(prefix="hero-fa4-operands-"))
    source_path = directory / f"{module_name.rsplit('.', 1)[-1]}.py"
    source_path.write_text(patched)
    module_spec = importlib.util.spec_from_file_location(module_name, source_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load isolated native source from {source_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module
