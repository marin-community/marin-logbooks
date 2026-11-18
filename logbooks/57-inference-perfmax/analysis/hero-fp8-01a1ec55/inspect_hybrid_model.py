# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require actual BF16 Triton and FP8 Marlin storage on every DP rank."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path


def main():
    profile = json.loads(Path(os.environ["KNOTMETER_PROFILE"]).read_text())
    output = Path(os.environ["KNOTMETER_CHECK_OUTPUT"])
    arguments = profile["serving"]["extra_args"]
    selector = json.loads(arguments[arguments.index("--quantization-config") + 1])
    backend = json.loads(arguments[arguments.index("--kernel-config") + 1])["moe_backend"]
    if profile["quantization"] != "online" or backend != "marlin" or selector.get("moe") != "fp8_per_tensor":
        raise SystemExit(
            "This metadata policy covers only existing per-tensor online FP8 with native Marlin and BF16 Triton"
        )
    ignore = selector.get("ignore", [])
    bf16_layers = [i for i in range(48) if f"model.layers.{i}.mlp.experts*" in ignore]
    if selector != {"moe": "fp8_per_tensor", **({"ignore": ignore} if ignore else {})} or ignore != [
        f"model.layers.{i}.mlp.experts*" for i in bf16_layers
    ]:
        raise SystemExit("Unknown online selector or ignored layer pattern")
    directory = Path(__file__).parent
    hashes = {
        "extension_sha256": hashlib.sha256((directory / "hero_hybrid_metadata.py").read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256((directory / "hero_weight_metadata.py").read_bytes()).hexdigest(),
    }
    body = {"method": "inspect_hero_hybrid_storage", "args": [bf16_layers, output.parent.name], "timeout": 120}
    request = urllib.request.Request(
        os.environ["KNOTMETER_SERVER_URL"].removesuffix("/v1") + "/collective_rpc",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    report = {
        "status": "failed",
        "runtime_source": os.environ["KNOTMETER_RUNTIME_SOURCE"],
        "expected_bf16_layers": bf16_layers,
        "expected_fp8_layers": [i for i in range(48) if i not in bf16_layers],
        "expected_weight_quantization": (
            "Native per-tensor FP8 E4M3FN selected experts; BF16 remaining experts/projections; FP32 router"
        ),
        "expected_activation_precision": "Native BF16 activations in both Marlin FP8 W8A16 and ignored BF16 Triton MoE",
        "expected_kv_precision": "BF16 attention and ShortConv state, verified separately from expert precision",
        "expected_hashes": hashes,
        "coverage": (
            "Pinned native DPLB RPC broadcasts to every DP engine; any failure propagates. "
            "Full per-rank metadata stays in each host archive."
        ),
    }
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            inspection = json.load(response)
        reports = inspection["results"]
        passed = bool(reports) and all(
            item["status"] == "passed"
            and all(item[key] == value for key, value in hashes.items())
            and item["dp_size"] == profile["gpus"]["count"]
            and item["tp_size"] == 1
            and item["expected_bf16_layers"] == bf16_layers
            and item["bf16_expert_layers"] == len(bf16_layers)
            and item["fp8_expert_layers"] == 48 - len(bf16_layers)
            for item in reports
        )
        report |= {"status": "passed" if passed else "failed", "inspection": inspection}
    except Exception as error:
        report["error"] = str(error)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise SystemExit("Actual mixed BF16/FP8 loaded storage did not pass")


if __name__ == "__main__":
    main()
