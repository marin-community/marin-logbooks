# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate KV head count and page size in a captured native decode call."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from compare_moe_snapshots import compare_tensor
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

ATOL = 1e-2
RTOL = 1e-2


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(exist_ok=False)
    captures = {}
    files = {}
    for label, directory in (("reference", args.reference), ("candidate", args.candidate)):
        captures[label] = []
        for ordinal in range(6):
            suffix = "" if ordinal == 0 else f"-prefill-{ordinal}"
            path = directory / f"attention-layer-3{suffix}.pt"
            capture = torch.load(path, map_location="cpu", weights_only=True)
            assert capture["layer"] == 3 and capture["step_ordinal"] == ordinal
            captures[label].append(capture)
            files[f"{label}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    left, right = captures["reference"], captures["candidate"]
    assert [c["num_actual_tokens"] for c in left] == [2048, 2048, 2048, 2048, 15, 1]
    assert left[-1]["num_decodes"] == right[-1]["num_decodes"] == 1
    query = left[-1]["query"].reshape(1, 48, 128).cuda()
    assert torch.equal(left[-1]["query"], right[-1]["query"])
    histories = {}
    for key in ("key", "value"):
        baseline = torch.cat([c[key].reshape(c["num_actual_tokens"], 12, 128) for c in left])
        logical = torch.cat([c[key].reshape(c["num_actual_tokens"], 6, 128) for c in right])
        assert torch.equal(baseline[:, ::2], baseline[:, 1::2])
        assert torch.equal(baseline[:, ::2], logical)
        histories[key] = logical.cuda()
    length = histories["key"].shape[0]
    head = torch.arange(48, device="cuda") // 8
    scores = torch.matmul(query.float().transpose(0, 1), histories["key"][:, head].float().permute(1, 2, 0))
    expected = torch.matmul(
        torch.softmax(scores * (128**-0.5), dim=-1), histories["value"][:, head].float().transpose(0, 1)
    ).transpose(0, 1)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    outputs = {}
    cases = []
    for disable_split in (False, True):
        for heads in (12, 6):
            for page_size in (16, 32):
                pages = (length + page_size - 1) // page_size
                packed = torch.zeros(pages, heads, page_size, 256, dtype=torch.bfloat16, device="cuda")
                for index, key in enumerate(("key", "value")):
                    history = histories[key].repeat_interleave(heads // 6, dim=1)
                    padded = torch.zeros(pages * page_size, heads, 128, dtype=history.dtype, device="cuda")
                    padded[:length].copy_(history)
                    packed[..., index * 128 : (index + 1) * 128].copy_(
                        padded.reshape(pages, page_size, heads, 128).transpose(1, 2)
                    )
                wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "HND", use_tensor_cores=True, backend="auto")
                wrapper.plan(
                    torch.tensor([0, pages], dtype=torch.int32),
                    torch.arange(pages, device="cuda", dtype=torch.int32),
                    torch.tensor([(length - 1) % page_size + 1], dtype=torch.int32),
                    num_qo_heads=48,
                    num_kv_heads=heads,
                    head_dim=128,
                    page_size=page_size,
                    pos_encoding_mode="NONE",
                    sm_scale=128**-0.5,
                    window_left=-1,
                    logits_soft_cap=0.0,
                    q_data_type=torch.bfloat16,
                    kv_data_type=torch.bfloat16,
                    o_data_type=torch.bfloat16,
                    fixed_split_size=2048 if disable_split else -1,
                    disable_split_kv=disable_split,
                )
                output = wrapper.run(query, packed.split(128, dim=-1))
                repeated = wrapper.run(query, packed.split(128, dim=-1))
                assert torch.equal(output, repeated)
                torch.testing.assert_close(output.float(), expected, atol=ATOL, rtol=RTOL)
                name = f"heads-{heads}-page-{page_size}-no-split-{disable_split}"
                outputs[name] = output.cpu()
                cases.append(
                    {
                        "name": name,
                        "backend": wrapper._backend,
                        "kv_heads": heads,
                        "page_size": page_size,
                        "disable_split_kv": disable_split,
                        "repeat_exact": True,
                        "native_tolerance_passed": True,
                        "fp32_max_absolute_delta": (output.float() - expected).abs().max().item(),
                        "captured_reference": compare_tensor(left[-1]["output"].reshape(1, 48, 128), output.cpu()),
                        "captured_candidate": compare_tensor(right[-1]["output"].reshape(1, 48, 128), output.cpu()),
                    }
                )
    torch.save({"outputs": outputs, "fp32_reference": expected.cpu()}, args.output / "outputs.pt")
    result = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "flashinfer_source_sha256": (
            hashlib.sha256(Path(inspect.getfile(BatchDecodeWithPagedKVCacheWrapper)).read_bytes()).hexdigest()
        ),
        "input_files": files,
        "context_length": length,
        "native_flashinfer_atol": ATOL,
        "native_flashinfer_rtol": RTOL,
        "cases": cases,
        "scope": "Captured global-layer decode component; fixed logical Q/K/V, no model or throughput qualification.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
