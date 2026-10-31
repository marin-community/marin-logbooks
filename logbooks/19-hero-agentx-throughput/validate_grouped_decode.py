# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the packaged grouped-decode path on ragged, offset KV histories."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from compare_moe_snapshots import compare_tensor
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from vllm import envs
from vllm.v1.attention.backends.grug_logical_kv import trtllm_decode_with_query_groups

ATOL = 1e-2
RTOL = 1e-2


def packed_history(histories: dict, heads: int, page_size: int) -> torch.Tensor:
    length = histories["key"].shape[0]
    pages = (length + page_size - 1) // page_size
    result = torch.zeros(pages, heads, page_size, 256, dtype=torch.bfloat16, device="cuda")
    for index, key in enumerate(("key", "value")):
        padded = torch.zeros(pages * page_size, heads, 128, dtype=torch.bfloat16, device="cuda")
        padded[:length].copy_(histories[key].repeat_interleave(heads // 6, dim=1))
        result[..., index * 128 : (index + 1) * 128].copy_(padded.reshape(pages, page_size, heads, 128).transpose(1, 2))
    return result


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
        captures.append(torch.load(path, map_location="cpu", weights_only=True))
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    query = captures[-1]["query"].reshape(1, 48, 128).cuda()
    source_histories = {}
    for key in ("key", "value"):
        full = torch.cat([c[key].reshape(c["num_actual_tokens"], 12, 128) for c in captures])
        assert torch.equal(full[:, ::2], full[:, 1::2])
        source_histories[key] = full[:, ::2].contiguous().cuda()
    workspace = torch.zeros(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda")
    head = torch.arange(48, device="cuda") // 8
    cases = []
    saved = {}
    for length in (2048, 8208, 262144):
        for batch in (1, 4, 32):
            physical_tokens = length + 32 * batch
            histories = {
                key: value.repeat((physical_tokens + value.shape[0] - 1) // value.shape[0], 1, 1)[:physical_tokens]
                for key, value in source_histories.items()
            }
            queries = torch.cat([query.roll(index % 48, dims=1) for index in range(batch)])
            lengths = [length - (index % 8) * 37 for index in range(batch)]
            expected_rows = []
            # Bound the independent FP32 scratch to one request, even at256K.
            for index, seq_len in enumerate(lengths):
                start = index * 32
                keys = histories["key"][start : start + seq_len, head].float().transpose(0, 1)
                scores = torch.matmul(queries[index].float()[:, None], keys.transpose(1, 2))
                del keys
                values = histories["value"][start : start + seq_len, head].float().transpose(0, 1)
                expected_rows.append(torch.matmul(torch.softmax(scores * (128**-0.5), dim=-1), values).squeeze(1))
                del values
            expected = torch.stack(expected_rows)
            outputs = {}
            for name, heads, page_size in (("original", 12, 16), ("grouped", 6, 32)):
                cache = packed_history(histories, heads, page_size)
                pages = (length + page_size - 1) // page_size
                block_tables = torch.arange(pages, device="cuda", dtype=torch.int32)[None] + (
                    torch.arange(batch, device="cuda", dtype=torch.int32)[:, None] * (32 // page_size)
                )
                call = {
                    "query": queries,
                    "kv_cache": cache.split(128, dim=-1),
                    "workspace_buffer": workspace,
                    "block_tables": block_tables,
                    "seq_lens": torch.tensor(lengths, device="cuda", dtype=torch.int32),
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
                output = torch.empty_like(queries)
                repeated = torch.empty_like(queries)
                if name == "grouped":
                    trtllm_decode_with_query_groups(**call, out=output, groups=2)
                    trtllm_decode_with_query_groups(**call, out=repeated, groups=2)
                else:
                    trtllm_batch_decode_with_kv_cache(**call, out=output)
                    trtllm_batch_decode_with_kv_cache(**call, out=repeated)
                assert torch.equal(output, repeated)
                torch.testing.assert_close(output.float(), expected, atol=ATOL, rtol=RTOL)
                if name == "original" and batch == 1 and length == 8208:
                    assert torch.equal(output.cpu(), captures[-1]["output"].reshape(1, 48, 128))
                outputs[name] = output.cpu()
                del cache, call
            name = f"ragged-batch-{batch}-max-length-{length}"
            cases.append(
                {
                    "name": name,
                    "batch": batch,
                    "seq_lens": lengths,
                    "token_offsets": [index * 32 for index in range(batch)],
                    "repeat_exact": True,
                    "native_fp32_tolerances_passed": True,
                    "source_comparison": compare_tensor(outputs["original"], outputs["grouped"]),
                }
            )
            saved[name] = outputs
            print("CASE_COMPLETE", name, cases[-1]["source_comparison"]["exact"], flush=True)
    distribution = importlib.metadata.distribution("vllm")
    result = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "installed_wrapper_sha256": (
            hashlib.sha256(
                Path(distribution.locate_file("vllm/v1/attention/backends/grug_logical_kv.py")).read_bytes()
            ).hexdigest()
        ),
        "input_files": files,
        "fp32_atol": ATOL,
        "fp32_rtol": RTOL,
        "all_source_comparisons_exact": all(case["source_comparison"]["exact"] for case in cases),
        "cases": cases,
        "scope": (
            "Actual packaged native decode wrapper, ragged histories and full256Kcontext; no full-model qualification."
        ),
    }
    torch.save(saved, args.output / "outputs.pt")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
