# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test GQA4 decode with six stored KV heads and partitioned query heads."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from compare_moe_snapshots import compare_tensor
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from vllm import envs

ATOL = 1e-2
RTOL = 1e-2


def partition_query(query: torch.Tensor) -> torch.Tensor:
    batch = query.shape[0]
    return query.reshape(batch, 6, 2, 4, 128).permute(0, 2, 1, 3, 4).reshape(batch * 2, 24, 128).contiguous()


def restore_query(output: torch.Tensor) -> torch.Tensor:
    batch = output.shape[0] // 2
    return output.reshape(batch, 2, 6, 4, 128).permute(0, 2, 1, 3, 4).reshape(batch, 48, 128)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(exist_ok=False)
    captures = []
    files = {}
    for ordinal in range(6):
        suffix = "" if ordinal == 0 else f"-prefill-{ordinal}"
        path = args.reference / f"attention-layer-3{suffix}.pt"
        capture = torch.load(path, map_location="cpu", weights_only=True)
        assert capture["layer"] == 3 and capture["step_ordinal"] == ordinal
        captures.append(capture)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert [c["num_actual_tokens"] for c in captures] == [2048, 2048, 2048, 2048, 15, 1]
    assert captures[-1]["num_decodes"] == 1
    query = captures[-1]["query"].reshape(1, 48, 128).cuda()
    assert torch.equal(restore_query(partition_query(query)), query)
    histories = {}
    for key in ("key", "value"):
        full = torch.cat([c[key].reshape(c["num_actual_tokens"], 12, 128) for c in captures])
        assert torch.equal(full[:, ::2], full[:, 1::2])
        histories[key] = full[:, ::2].contiguous().cuda()
    length = histories["key"].shape[0]
    head = torch.arange(48, device="cuda") // 8
    scores = torch.matmul(query.float().transpose(0, 1), histories["key"][:, head].float().permute(1, 2, 0))
    expected = torch.matmul(
        torch.softmax(scores * (128**-0.5), dim=-1), histories["value"][:, head].float().transpose(0, 1)
    ).transpose(0, 1)
    workspace = torch.zeros(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda")
    outputs = {}
    cases = []
    for page_size in (16, 32):
        pages = (length + page_size - 1) // page_size
        for mode in ("original-12kv", "logical-6kv", "batched-gqa4", "separate-gqa4"):
            heads = 12 if mode == "original-12kv" else 6
            packed = torch.zeros(pages, heads, page_size, 256, dtype=torch.bfloat16, device="cuda")
            for index, key in enumerate(("key", "value")):
                history = histories[key].repeat_interleave(heads // 6, dim=1)
                padded = torch.zeros(pages * page_size, heads, 128, dtype=history.dtype, device="cuda")
                padded[:length].copy_(history)
                packed[..., index * 128 : (index + 1) * 128].copy_(
                    padded.reshape(pages, page_size, heads, 128).transpose(1, 2)
                )
            block_table = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
            seq_lens = torch.tensor([length], dtype=torch.int32, device="cuda")
            call = {
                "kv_cache": packed.split(128, dim=-1),
                "workspace_buffer": workspace,
                "max_seq_len": length,
                "bmm1_scale": 128**-0.5,
                "bmm2_scale": 1.0,
                "window_left": -1,
                "kv_layout": "HND",
                "backend": "trtllm-gen",
                "q_len_per_req": 1,
            }
            repeats = []
            for _ in range(2):
                if mode == "batched-gqa4":
                    grouped = partition_query(query)
                    output = torch.empty_like(grouped)
                    trtllm_batch_decode_with_kv_cache(
                        **call,
                        query=grouped,
                        out=output,
                        block_tables=block_table.repeat_interleave(2, dim=0),
                        seq_lens=seq_lens.repeat_interleave(2),
                    )
                    output = restore_query(output)
                elif mode == "separate-gqa4":
                    grouped = partition_query(query)
                    output = torch.empty_like(grouped)
                    for group in range(2):
                        trtllm_batch_decode_with_kv_cache(
                            **call,
                            query=grouped[group : group + 1],
                            out=output[group : group + 1],
                            block_tables=block_table,
                            seq_lens=seq_lens,
                        )
                    output = restore_query(output)
                else:
                    output = torch.empty_like(query)
                    trtllm_batch_decode_with_kv_cache(
                        **call, query=query, out=output, block_tables=block_table, seq_lens=seq_lens
                    )
                repeats.append(output)
            assert torch.equal(*repeats)
            output = repeats[0]
            torch.testing.assert_close(output.float(), expected, atol=ATOL, rtol=RTOL)
            name = f"{mode}-page-{page_size}"
            baseline = captures[-1]["output"].reshape(1, 48, 128)
            if mode == "original-12kv":
                assert torch.equal(output.cpu(), baseline), "Native baseline must reproduce its captured output"
            outputs[name] = output.cpu()
            cases.append(
                {
                    "name": name,
                    "mode": mode,
                    "kv_heads": heads,
                    "page_size": page_size,
                    "repeat_exact": True,
                    "native_tolerance_passed": True,
                    "fp32_max_absolute_delta": (output.float() - expected).abs().max().item(),
                    "captured_reference": compare_tensor(baseline, output.cpu()),
                    "kv_storage_bytes": packed.numel() * packed.element_size(),
                }
            )
    torch.save({"outputs": outputs, "fp32_reference": expected.cpu()}, args.output / "outputs.pt")
    result = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_decode_source_sha256": (
            hashlib.sha256(Path(inspect.getfile(trtllm_batch_decode_with_kv_cache)).read_bytes()).hexdigest()
        ),
        "input_files": files,
        "context_length": length,
        "native_flashinfer_atol": ATOL,
        "native_flashinfer_rtol": RTOL,
        "cases": cases,
        "scope": "Captured global-layer decode component only. No model, source-gate or throughput qualification.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
