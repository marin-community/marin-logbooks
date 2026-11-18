# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain the loaded model's native quantization inspection before measurement."""

import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    profile = json.loads(Path(os.environ["KNOTMETER_PROFILE"]).read_text())
    base = os.environ["KNOTMETER_SERVER_URL"].removesuffix("/v1")
    request = urllib.request.Request(
        f"{base}/collective_rpc",
        data=json.dumps({"method": "get_model_inspection", "timeout": 120}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        inspection = json.load(response)
    text = "\n".join(inspection["results"])
    expected = profile["quantization"]
    observed = {
        "online_fp8_linear": "Fp8PerTensorOnlineLinearMethod" in text,
        "online_fp8_experts": "Fp8PerTensorOnlineMoEMethod" in text,
    }
    if expected == "fp8":
        passed = all(observed.values())
    elif expected == "online":
        arguments = profile["serving"]["extra_args"]
        selector = json.loads(arguments[arguments.index("--quantization-config") + 1])
        passed = selector == {"moe": "fp8_per_tensor"} and observed == {
            "online_fp8_linear": False,
            "online_fp8_experts": True,
        }
    else:
        passed = not any(observed.values())
    report = {
        "status": "passed" if passed else "failed",
        "runtime_source": os.environ["KNOTMETER_RUNTIME_SOURCE"],
        "expected_weight_quantization": expected,
        "expected_kv_dtype": "auto (bfloat16 model dtype)",
        "observed": observed,
        "inspection": inspection,
        "scope": "Native post-load quantization-method inspection; no tensor values are read.",
    }
    Path(os.environ["KNOTMETER_CHECK_OUTPUT"]).write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("The loaded model's quantization methods differ from the profile")


if __name__ == "__main__":
    main()
