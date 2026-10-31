# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check batched GQA4 decode across request counts and history lengths."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from compare_moe_snapshots import compare_tensor
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from replay_attention_head_groups import ATOL, RTOL, partition_query, restore_query
from vllm import envs


def pack_history(histories: dict[str, torch.Tensor], length: int, heads: int, page_size: int) -> torch.Tensor:
    pages = (length + page_size - 1) // page_size
    packed = torch.zeros(pages, heads, page_size, 256, dtype=torch.bfloat16, device="cuda")
    for index, key in enumerate(("key", "value")):
        padded = torch.zeros(pages * page_size, heads, 128, dtype=torch.bfloat16, device="cuda")
        padded[:length].copy_(histories[key][:length].repeat_interleave(heads // 6, dim=1))
        packed[..., index * 128 : (index + 1) * 128].copy_(padded.reshape(pages, page_size, heads, 128).transpose(1, 2))
    return packed


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-query-groups", action="store_true")
    args = parser.parse_args()
    if args.native_query_groups:
        from vllm.v1.attention.backends.grug_logical_kv import trtllm_decode_with_query_groups
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(exist_ok=False)
    captures = []
    for ordinal in range(6):
        suffix = "" if ordinal == 0 else f"-prefill-{ordinal}"
        captures.append(torch.load(args.reference / f"attention-layer-3{suffix}.pt", weights_only=True))
    query = captures[-1]["query"].reshape(1, 48, 128).cuda()
    histories = {}
    for key in ("key", "value"):
        full = torch.cat([c[key].reshape(c["num_actual_tokens"], 12, 128) for c in captures])
        assert torch.equal(full[:, ::2], full[:, 1::2])
        histories[key] = full[:, ::2].contiguous().cuda()
    workspace = torch.zeros(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda")
    head = torch.arange(48, device="cuda") // 8
    cases = []
    saved = {}
    for length in (2048, 8208):
        caches = {"original": pack_history(histories, length, 12, 16), "grouped": pack_history(histories, length, 6, 32)}
        for batch in (1, 4, 16, 32):
            queries = torch.cat([query.roll(index % 48, dims=1) for index in range(batch)])
            assert torch.equal(restore_query(partition_query(queries)), queries)
            scores = torch.einsum("bhd,lhd->bhl", queries.float(), histories["key"][:length, head].float())
            expected = torch.einsum(
                "bhl,lhd->bhd", torch.softmax(scores * (128**-0.5), dim=-1), histories["value"][:length, head].float()
            )
            outputs = {}
            for mode, page_size in (("original", 16), ("grouped", 32)):
                pages = (length + page_size - 1) // page_size
                native_grouped = args.native_query_groups and mode == "grouped"
                manual_grouped = mode == "grouped" and not native_grouped
                effective_batch = batch * (2 if manual_grouped else 1)
                q = partition_query(queries) if manual_grouped else queries
                call = {
                    "query": q,
                    "kv_cache": caches[mode].split(128, dim=-1),
                    "workspace_buffer": workspace,
                    "block_tables": (
                        torch.arange(pages, dtype=torch.int32, device="cuda")[None].repeat(effective_batch, 1)
                    ),
                    "seq_lens": torch.full((effective_batch,), length, dtype=torch.int32, device="cuda"),
                    "max_seq_len": length,
                    "bmm1_scale": 128**-0.5,
                    "bmm2_scale": 1.0,
                    "window_left": -1,
                    "kv_layout": "HND",
                    "backend": "trtllm-gen",
                    "q_len_per_req": 1,
                    "max_q_len": None,
                    "cum_seq_lens_q": None,
                    "sinks": None,
                    "o_sf_scale": None,
                    "kv_cache_sf": None,
                    "lse": None,
                    "return_lse": False,
                }
                output = torch.empty_like(q)
                repeated = torch.empty_like(q)
                if native_grouped:
                    trtllm_decode_with_query_groups(**call, out=output, groups=2)
                    trtllm_decode_with_query_groups(**call, out=repeated, groups=2)
                else:
                    trtllm_batch_decode_with_kv_cache(**call, out=output)
                    trtllm_batch_decode_with_kv_cache(**call, out=repeated)
                assert torch.equal(output, repeated)
                output = restore_query(output) if manual_grouped else output
                torch.testing.assert_close(output.float(), expected, atol=ATOL, rtol=RTOL)
                outputs[mode] = output.cpu()
            name = f"batch-{batch}-length-{length}"
            cases.append(
                {
                    "name": name,
                    "batch": batch,
                    "length": length,
                    "original_page_size": 16,
                    "grouped_page_size": 32,
                    "repeat_exact": True,
                    "native_fp32_tolerances_passed": True,
                    "source_comparison": compare_tensor(outputs["original"], outputs["grouped"]),
                }
            )
            saved[name] = outputs
    torch.save(saved, args.output / "outputs.pt")
    result = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "query_group_source_sha256": (
            hashlib.sha256(Path(__file__).with_name("replay_attention_head_groups.py").read_bytes()).hexdigest()
        ),
        "fp32_atol": ATOL,
        "fp32_rtol": RTOL,
        "native_query_group_wrapper": args.native_query_groups,
        "cases": cases,
        "scope": "Captured-history component grid. No full-model, mixed prefill/decode or throughput qualification.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
