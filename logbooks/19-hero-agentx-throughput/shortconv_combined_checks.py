# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native regressions loaded by the existing single-rank probe driver.

History/mixed-prefill checks come unchanged from PR107. The PR108 padded-row
check uses Hero's full width of 6,144 instead of its small width of two.
"""

from dataclasses import replace

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.grugmoe import GrugMoeShortConv
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata


def _causal_depthwise_reference(
    sequence: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(sequence)
    for token_index in range(sequence.shape[0]):
        for lag in range(weight.shape[0]):
            if token_index >= lag:
                output[token_index] += weight[lag] * sequence[token_index - lag]
    return output


def test_hero_grug_moe_sconv_preserves_request_history_and_isolation():
    prefix = "hero.sconv.history"
    module = GrugMoeShortConv(
        dim=2,
        kernel_size=4,
        params_dtype=torch.float32,
        cache_config=None,
        prefix=prefix,
    ).cuda()
    weight = torch.tensor(
        [[1.0, 0.75], [0.5, -0.25], [-0.25, 0.125], [0.125, 0.5]],
        device="cuda",
    )
    module.weight.data.copy_(weight)
    # State slot zero is vLLM's reserved null block.
    module.kv_cache = (torch.zeros(3, 3, 2, device="cuda"),)

    request_a = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], device="cuda")
    request_b = torch.tensor([[10.0, 4.0], [20.0, 8.0]], device="cuda")
    prefill = torch.vstack((request_a, request_b))
    for profiling_metadata in (None, {}):
        # V1 omits all metadata; V2's profiling pass omits Mamba groups from
        # its dictionary. Neither pass may consume or mutate request history.
        with set_forward_context(
            profiling_metadata,
            get_current_vllm_config(),
            num_tokens=prefill.shape[0],
        ):
            profiling_output = module(prefill)
        torch.testing.assert_close(profiling_output, prefill * weight[0], rtol=0, atol=0)
        torch.testing.assert_close(module.kv_cache[0], torch.zeros_like(module.kv_cache[0]), rtol=0, atol=0)
    prefill_metadata = ShortConvAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=5,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=2,
        query_start_loc_p=torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
        has_initial_states_p=torch.tensor([False, False], device="cuda"),
        state_indices_tensor_p=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.empty(0, device="cuda", dtype=torch.int32),
        num_accepted_tokens=None,
        query_start_loc_d=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([3, 2], device="cuda"),
    )
    with set_forward_context(
        {prefix: prefill_metadata},
        get_current_vllm_config(),
        num_tokens=prefill.shape[0],
    ):
        prefill_output = module(prefill)

    expected_a = _causal_depthwise_reference(request_a, weight)
    expected_b = _causal_depthwise_reference(request_b, weight)
    torch.testing.assert_close(
        prefill_output,
        torch.vstack((expected_a, expected_b)),
    )

    decode = torch.tensor([[4.0, -4.0], [30.0, 12.0]], device="cuda")
    decode_metadata = ShortConvAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_reqs=2,
        query_start_loc_p=None,
        has_initial_states_p=None,
        state_indices_tensor_p=torch.empty(0, device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        num_accepted_tokens=None,
        query_start_loc_d=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([4, 3], device="cuda"),
    )
    with set_forward_context(
        {prefix: decode_metadata},
        get_current_vllm_config(),
        num_tokens=decode.shape[0],
    ):
        decode_output = module(decode)

    expected_decode = torch.vstack(
        (
            _causal_depthwise_reference(torch.vstack((request_a, decode[:1])), weight)[-1:],
            _causal_depthwise_reference(torch.vstack((request_b, decode[1:])), weight)[-1:],
        )
    )
    torch.testing.assert_close(decode_output, expected_decode)

    # Decode A while B resumes prefill from its existing state. The output
    # keeps decode rows first even though the prefill kernel executes first.
    next_a = torch.tensor([[5.0, -5.0]], device="cuda")
    next_b = torch.tensor([[40.0, 16.0], [50.0, 20.0]], device="cuda")
    mixed = torch.vstack((next_a, next_b))
    mixed_metadata = replace(
        prefill_metadata,
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=1,
        num_decode_tokens=1,
        query_start_loc_p=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        has_initial_states_p=torch.tensor([True], device="cuda"),
        state_indices_tensor_p=torch.tensor([2], device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.tensor([1], device="cuda", dtype=torch.int32),
        query_start_loc_d=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor([5, 5], device="cuda"),
    )
    with set_forward_context(
        {prefix: mixed_metadata},
        get_current_vllm_config(),
        num_tokens=mixed.shape[0],
    ):
        mixed_output = module(mixed)

    expected_mixed = torch.vstack(
        (
            _causal_depthwise_reference(torch.vstack((request_a, decode[:1], next_a)), weight)[-1:],
            _causal_depthwise_reference(torch.vstack((request_b, decode[1:], next_b)), weight)[-2:],
        )
    )
    torch.testing.assert_close(mixed_output, expected_mixed)


def test_hero_grug_moe_sconv_padded_decode_zeros_inactive_rows():
    prefix = "hero.sconv.padding"
    module = GrugMoeShortConv(6144, 4, torch.bfloat16, None, prefix).cuda()
    module.weight.data.fill_(1)
    module.kv_cache = (torch.zeros(2, 3, 6144, device="cuda", dtype=torch.bfloat16),)
    metadata = ShortConvAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_reqs=1,
        query_start_loc_p=None,
        has_initial_states_p=None,
        state_indices_tensor_p=torch.empty(0, device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.tensor([1], device="cuda", dtype=torch.int32),
        num_accepted_tokens=None,
        query_start_loc_d=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([1], device="cuda"),
    )
    for value, expected in ((2.0, 2.0), (3.0, 5.0)):
        hidden = torch.full((4, 6144), float("nan"), device="cuda", dtype=torch.bfloat16)
        hidden[0].fill_(value)
        output = torch.full_like(hidden, float("nan"))
        with set_forward_context({prefix: metadata}, get_current_vllm_config(), num_tokens=4):
            torch.ops.vllm.grug_moe_short_conv(hidden, output, prefix)
        torch.testing.assert_close(output[0], torch.full_like(output[0], expected), rtol=0, atol=0)
        torch.testing.assert_close(output[1:], torch.zeros_like(output[1:]), rtol=0, atol=0)
