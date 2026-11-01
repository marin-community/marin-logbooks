# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare qualified native expert operators at local24/latent3072 geometry."""

import argparse
import json
import os
import sys
from pathlib import Path

from experiments.evaluation.agentx.processes import run_logged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-source-sha256", required=True)
    parser.add_argument("--timing-repeats", type=int, default=20)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=False)
    results = {}
    for backend in ("triton", "flashinfer_cutlass"):
        output = args.output.parent / backend / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("moe_output_probe.py")),
            "--output",
            str(output),
            "--model-source-sha256",
            args.model_source_sha256,
            "--backend",
            backend,
            "--hidden-dim",
            "3072",
            "--intermediate-dim",
            "3072",
            "--num-experts",
            "24",
            "--top-k",
            "8",
            "--normal-weights",
            "--normal-inputs",
            "--native-only",
            "--reference-chunk-rows",
            "4",
            "--timing-repeats",
            str(args.timing_repeats),
            "--row-counts",
            "1",
            "3",
            "8",
            "17",
            "32",
            "64",
            "256",
            "2048",
            "2",
            "1",
        ]
        run_logged(command, dict(os.environ), args.output.parent / (backend + ".log"), timeout=900)
        result = json.loads(output.read_text())
        assert result["status"] == "native_backend_existing_torch_tolerances_passed"
        assert result["enable_flashinfer_autotune"] is False
        results[backend] = result
    comparisons = []
    for reference, candidate in zip(results["triton"]["rows"], results["flashinfer_cutlass"]["rows"], strict=True):
        assert reference["token_rows"] == candidate["token_rows"]
        baseline = reference["timing"]["median_milliseconds"]
        alternative = candidate["timing"]["median_milliseconds"]
        comparisons.append(
            {
                "token_rows": reference["token_rows"],
                "triton_median_milliseconds": baseline,
                "cutlass_median_milliseconds": alternative,
                "triton_over_cutlass_latency_ratio": baseline / alternative,
            }
        )
    result = {
        "status": "both_native_component_numerics_and_timings_complete",
        "results": results,
        "comparisons": comparisons,
        "scope": (
            "Seed 42 synthetic BF16 native expert operators: 24 local experts, top 8, latent dimension 3072. "
            "One visible GPU. "
            "Every timed output passes original FP32 absolute and relative tolerances 0.02. "
            "Input cloning and router linear are excluded; "
            "native host-dispatch GPU queue gaps can remain. Full 384-expert EP16, two shared experts, model-state "
            "equivalence and AgentX performance are unqualified. Ratios guide whether a full-model trial is useful."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
