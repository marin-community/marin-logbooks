# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research-only IID capture through vLLM's documented logits-processor API.

The pinned V1 sampler invokes argmax-invariant processors after temperature
and before top-k/top-p. This collector therefore rejects sampling filters.
It changes no logits and uses a separate RNG from the sampled action.
"""

import numpy as np
import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, MoveDirectionality

CAPTURE_KEY = "sc_sampled_capture_key"
_CAPTURES: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}


class SampledScoreCapture(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        if not (sampling_params.extra_args or {}).get(CAPTURE_KEY):
            return
        if sampling_params.temperature <= 0 or sampling_params.top_k != -1 or sampling_params.top_p != 1:
            raise ValueError("IID correction capture requires positive temperature, top_k=-1 and top_p=1")
        if sampling_params.min_p != 0:
            raise ValueError("IID correction capture precedes min-p filtering and requires min_p=0")

    def __init__(self, vllm_config, device, is_pin_memory):
        self.rows = {}
        self.generator = torch.Generator(device=device).manual_seed(1397)

    def is_argmax_invariant(self):
        return True

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.rows.pop(index, None)
        for index, params, prompt, output in batch_update.added:
            self.rows.pop(index, None)
            key = (params.extra_args or {}).get(CAPTURE_KEY)
            if key:
                self.rows[index] = (str(key), len(prompt or []), output)
        for source, target, direction in batch_update.moved:
            source_value, target_value = self.rows.pop(source, None), self.rows.pop(target, None)
            if source_value is not None:
                self.rows[target] = source_value
            if direction is MoveDirectionality.SWAP and target_value is not None:
                self.rows[source] = target_value

    def apply(self, logits):
        if not self.rows:
            return logits
        rows = list(self.rows)
        logprobs = logits[rows].log_softmax(-1, dtype=torch.float32)
        draws = torch.multinomial(logprobs.exp(), 8, replacement=True, generator=self.generator)
        scores = logprobs.gather(-1, draws)
        for offset, index in enumerate(rows):
            key, prompt_length, output = self.rows[index]
            _CAPTURES.setdefault(key, []).append(
                (prompt_length + len(output), draws[offset].detach(), scores[offset].detach())
            )
        return logits


def pop_captures(key: str):
    return [
        {
            "prefix_tokens": position,
            "indices": ids.cpu().numpy().astype(np.int32),
            "logprobs": scores.cpu().numpy(),
        }
        for position, ids, scores in _CAPTURES.pop(key, [])
    ]
