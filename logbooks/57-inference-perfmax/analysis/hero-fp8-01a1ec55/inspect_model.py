# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain the loaded model's native quantization inspection before measurement."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    profile = json.loads(Path(os.environ["KNOTMETER_PROFILE"]).read_text())
    base = os.environ["KNOTMETER_SERVER_URL"].removesuffix("/v1")
    output = Path(os.environ["KNOTMETER_CHECK_OUTPUT"])
    if profile["quantization"] == "online":
        inspect_storage(profile, base, output)
        return
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
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("The loaded model's quantization methods differ from the profile")


def inspect_storage(profile: dict, base: str, output: Path) -> None:
    arguments = profile["serving"]["extra_args"]
    selector = json.loads(arguments[arguments.index("--quantization-config") + 1])
    if selector != {"moe": "fp8_per_tensor"}:
        raise SystemExit("The storage policy does not cover this online selector")
    expected_hash = hashlib.sha256(Path(__file__).with_name("hero_weight_metadata.py").read_bytes()).hexdigest()
    body = {
        "method": "inspect_hero_storage",
        "args": ["selective-fp8-marlin", output.parent.name],
        "timeout": 120,
    }
    report = {
        "status": "failed",
        "runtime_source": os.environ["KNOTMETER_RUNTIME_SOURCE"],
        "expected_weight_quantization": "FP8 E4M3FN experts, BF16 projections, FP32 router",
        "expected_extension_sha256": expected_hash,
        "coverage": (
            "Pinned DPLB utility broadcasts to every DP engine and propagates any worker failure; "
            "Only the first result is returned. Every rank writes full metadata "
            "under weight-storage in its host archive."
        ),
    }
    try:
        request = urllib.request.Request(
            f"{base}/collective_rpc", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            inspection = json.load(response)
        reports = inspection["results"]
        passed = bool(reports) and all(
            item["status"] == "passed"
            and item["extension_sha256"] == expected_hash
            and item["dp_size"] == profile["gpus"]["count"]
            and item["tp_size"] == 1
            for item in reports
        )
        report |= {"status": "passed" if passed else "failed", "inspection": inspection}
    except Exception as error:
        report["error"] = str(error)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise SystemExit("Loaded weight-storage metadata did not pass")


if __name__ == "__main__":
    main()
