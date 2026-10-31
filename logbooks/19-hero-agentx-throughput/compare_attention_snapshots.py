# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare first native attention inputs/outputs and independent FP32 math."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from compare_moe_snapshots import compare_tensor

ATOL = 1e-2
RTOL = 1e-2


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefills", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = []
    for index in (0, 3):
        history = {"reference": [], "candidate": []}
        for ordinal in range(args.prefills):
            captures = {}
            files = {}
            for label, directory in [("reference", args.reference), ("candidate", args.candidate)]:
                suffix = "" if ordinal == 0 else f"-prefill-{ordinal}"
                path = directory / f"attention-layer-{index}{suffix}.pt"
                capture = torch.load(path, weights_only=True, map_location="cpu")
                assert capture["layer"] == index and capture["num_actual_tokens"] == 2048
                captures[label] = capture
                if args.prefills == 2:
                    assert capture["causal"] and capture["num_prefills"] == 1 and capture["num_decodes"] == 0
                    assert capture["prefill_metadata"]["seq_lens"].tolist() == [2048 * (ordinal + 1)]
                files[label] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            left, right = captures["reference"], captures["candidate"]
            comparisons = {key: compare_tensor(left[key], right[key]) for key in ("query", "output")}
            duplicate_heads = {}
            for key in ("key", "value"):
                a, b = left[key].reshape(2048, -1, 128), right[key].reshape(2048, -1, 128)
                if index == 3:
                    duplicate_heads[key] = bool(torch.equal(a[:, ::2], a[:, 1::2]))
                    a = a[:, ::2]
                comparisons[key] = compare_tensor(a, b)
            native_references = {}
            for label, capture in captures.items():
                q = capture["query"].reshape(2048, 48, 128).cuda().float().transpose(0, 1)
                history[label].append(capture)
                k = torch.cat([c["key"].reshape(2048, -1, 128) for c in history[label]]).cuda().float()
                v = torch.cat([c["value"].reshape(2048, -1, 128) for c in history[label]]).cuda().float()
                head = torch.arange(48, device="cuda") // (48 // k.shape[1])
                scores = torch.matmul(q, k[:, head].permute(1, 2, 0)) * (128**-0.5)
                end = torch.arange(2048, device="cuda")[:, None] + ordinal * 2048
                position = torch.arange(k.shape[0], device="cuda")[None, :]
                future = position > end
                if index == 0:
                    future |= position < end - 2048 + 1
                scores.masked_fill_(future, float("-inf"))
                expected = torch.matmul(torch.softmax(scores, dim=-1), v[:, head].transpose(0, 1)).transpose(0, 1)
                actual = capture["output"].reshape(2048, 48, 128).cuda().float()
                delta = (expected - actual).abs()
                close = torch.isclose(actual, expected, atol=ATOL, rtol=RTOL)
                native_references[label] = {
                    "native_flashinfer_tolerance_passed": bool(close.all()),
                    "different_elements_outside_tolerance": int((~close).sum()),
                    "max_absolute_delta": delta.max().item(),
                    "mean_absolute_delta": delta.mean().item(),
                    "cache_shape": capture["cache_shape"],
                }
                del q, k, v, scores, expected, actual, delta, close, future
            rows.append(
                {
                    "layer": index,
                    "prefill_ordinal": ordinal,
                    "files": files,
                    "cross_source": comparisons,
                    "baseline_duplicate_heads_exact": duplicate_heads,
                    "fp32_reference": native_references,
                }
            )
    result = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "native_attention_atol": ATOL,
        "native_attention_rtol": RTOL,
        "reference_precision": "FP32; TF32 disabled; consecutive cold2048-token causal/SWA prefills",
        "scope": "Actual native backend tensors on the synthetic geometry fixture. No full-Hero or speed gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
