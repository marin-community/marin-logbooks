# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject a BF16 backing store disguised by method names or an int32 view."""

import torch
from hero_weight_metadata import check_packed_fp8, describe_tensor


def test_real_storage_size_distinguishes_fp8_packing_from_bf16_reinterpretation():
    shape = (2, 32, 16)
    fp8 = torch.zeros(shape, dtype=torch.float8_e4m3fn)
    packed = fp8.view(torch.int32)
    logical = fp8.numel()
    assert check_packed_fp8(describe_tensor(packed), logical)
    assert not check_packed_fp8(describe_tensor(fp8), logical)

    # The same dtype and method name would be insufficient. A BF16 source
    # reinterpreted as int32 still occupies two bytes per logical weight.
    bf16 = torch.zeros(shape, dtype=torch.bfloat16).view(torch.int32)
    assert not check_packed_fp8(describe_tensor(bf16), logical)
    assert describe_tensor(bf16)["storage_bytes"] == 2 * logical
