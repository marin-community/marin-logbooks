# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check installed FA4 paged attention at the pinned Hero head geometry.

This one-GPU component check precedes a whole-model backend trial. Independent
FP32 selected-row references use the unchanged native BF16 attention tolerances.
It establishes neither the native metadata builder nor a serving speed gain.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import torch
from vllm.vllm_flash_attn import fa_version_unsupported_reason, flash_attn_varlen_func, is_fa_version_supported
from vllm.vllm_flash_attn.cute import interface

CASES = (
    ((1, 262144),),
    ((1, 1), (1, 16), (1, 2048), (1, 8192), (1, 16384), (1, 32768), (1, 65536), (1, 131072)),
    ((17, 8192), (64, 16384), (1, 65536)),
    ((2048, 65536),),
)
ATOL = 1.5e-2
RTOL = 1e-2


def selected_reference(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, table: torch.Tensor, lengths: tuple, window: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute FP32 attention before the candidate runs, including causal/SWA masks."""
    head_ids = torch.tensor([0, 3, 7, 12, 23, 35, 47], device=query.device)
    references, query_ids = [], []
    offset = 0
    for sequence, (query_len, kv_len) in enumerate(lengths):
        rows = torch.linspace(0, query_len - 1, min(8, query_len), device=query.device).long().unique()
        pages = table[sequence, : (kv_len + 15) // 16].long()
        keys = key[pages].reshape(-1, 12, 128)[:kv_len]
        values = value[pages].reshape(-1, 12, 128)[:kv_len]
        q = query[offset + rows][:, head_ids].float().transpose(0, 1)
        k = keys[:, head_ids // 4].float().permute(1, 2, 0)
        v = values[:, head_ids // 4].float().transpose(0, 1)
        scores = torch.matmul(q, k) * (128**-0.5)
        positions = torch.arange(kv_len, device=query.device)
        end = kv_len - query_len + rows
        mask = positions[None, :] > end[:, None]
        if window is not None:
            mask |= positions[None, :] < end[:, None] - window + 1
        scores.masked_fill_(mask[None], float("-inf"))
        references.append(torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1))
        query_ids.append(offset + rows)
        offset += query_len
    return torch.cat(references), torch.cat(query_ids), head_ids


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface-sha256", required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    assert hashlib.sha256(Path(interface.__file__).read_bytes()).hexdigest() == args.interface_sha256
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "starting",
        "attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interface_sha256": args.interface_sha256,
        "versions": {name: importlib.metadata.version(name) for name in ["vllm", "torch", "nvidia-cutlass-dsl"]},
        "geometry": {"query_heads": 48, "stored_kv_heads": 12, "head_size": 128, "page_size": 16},
        "reference_atol": ATOL,
        "reference_rtol": RTOL,
        "cases": [],
        "scope": "One-GPU synthetic FA4 component. No whole-model, metadata-builder or throughput qualification.",
    }
    path = args.output / "result.json"
    try:
        record["device"] = torch.cuda.get_device_name()
        record["capability"] = torch.cuda.get_device_capability()
        record["native_fa4_supported"] = is_fa_version_supported(4)
        record["native_fa4_unsupported_reason"] = fa_version_unsupported_reason(4)
        assert record["native_fa4_supported"], record["native_fa4_unsupported_reason"]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        for window in (None, 2048):
            for index, lengths in enumerate(CASES):
                torch.manual_seed(42 + index)
                pages_per_sequence = [(kv + 15) // 16 for _, kv in lengths]
                blocks = sum(pages_per_sequence)
                query = torch.randn(sum(q for q, _ in lengths), 48, 128, device="cuda", dtype=torch.bfloat16)
                key = torch.randn(blocks, 16, 12, 128, device="cuda", dtype=torch.bfloat16) / (128**0.5)
                value = torch.randn_like(key) / (128**0.5)
                if window is None:
                    # Global layers store six logical heads duplicated into twelve slots.
                    key.copy_(key[:, :, :6].repeat_interleave(2, dim=2))
                    value.copy_(value[:, :, :6].repeat_interleave(2, dim=2))
                table = torch.zeros(len(lengths), max(pages_per_sequence), device="cuda", dtype=torch.int32)
                permutation = torch.randperm(blocks, device="cuda", dtype=torch.int32)
                offset = 0
                for sequence, count in enumerate(pages_per_sequence):
                    table[sequence, :count] = permutation[offset : offset + count]
                    offset += count
                reference, query_ids, head_ids = selected_reference(query, key, value, table, lengths, window)
                cu_q = torch.tensor([0, *[q for q, _ in lengths]], device="cuda", dtype=torch.int32).cumsum(
                    0, dtype=torch.int32
                )
                output = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=cu_q,
                    seqused_k=torch.tensor([kv for _, kv in lengths], device="cuda", dtype=torch.int32),
                    max_seqlen_q=max(q for q, _ in lengths),
                    max_seqlen_k=max(pages_per_sequence) * 16,
                    softmax_scale=128**-0.5,
                    causal=True,
                    window_size=(-1, -1) if window is None else (window - 1, 0),
                    block_table=table,
                    num_splits=1,
                    fa_version=4,
                )
                assert output.dtype == torch.bfloat16 and output.shape == query.shape
                assert torch.isfinite(output).all()
                selected = output[query_ids][:, head_ids].float()
                torch.testing.assert_close(selected, reference, atol=ATOL, rtol=RTOL)
                label = f"window-{window}-case-{index}"
                torch.save(
                    {
                        "selected_output": selected.cpu(),
                        "fp32_reference": reference.cpu(),
                        "query_ids": query_ids.cpu(),
                        "head_ids": head_ids.cpu(),
                    },
                    args.output / (label + ".pt"),
                )
                record["cases"].append(
                    {
                        "label": label,
                        "seed": 42 + index,
                        "sequence_lengths": lengths,
                        "selected_values": selected.numel(),
                        "max_absolute_delta": (selected - reference).abs().max().item(),
                        "mean_absolute_delta": (selected - reference).abs().mean().item(),
                        "finite": True,
                        "native_reference_tolerance_passed": True,
                    }
                )
                path.write_text(json.dumps(record, indent=2) + "\n")
                del query, key, value, table, output, reference, selected
        record["status"] = "all_eight_native_tolerance_checks_passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
