# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repeat native tiny-model runs before attributing numerical drift to graphs."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from cache_probe import compare

from experiments.evaluation.agentx.artifacts import copy_file, unpack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.manifest.read_text())
    assert os.environ["IRIS_NODE_NAME"] in config["placement"]["nodes"]
    identity = {key: os.environ[key] for key in ("IRIS_NODE_NAME", "IRIS_TASK_ID", "IRIS_ATTEMPT_UID", "IRIS_BUNDLE_ID")}
    identity["gpus"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True
    ).splitlines()
    (args.output / "hardware-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    inputs = Path("/iris/candidates/tiny-aa-inputs")
    inputs.mkdir(parents=True, exist_ok=False)
    for key in ("wheel", "fixture"):
        source = config["inputs"][key]
        path = inputs / source["uri"].rsplit("/", 1)[1]
        copy_file(source["uri"], str(path))
        assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == source["sha256"]
    fixture = inputs / "fixture"
    unpack(inputs / config["inputs"]["fixture"]["uri"].rsplit("/", 1)[1], fixture)
    assert hashlib.sha256((fixture / "model.safetensors").read_bytes()).hexdigest() == config["fixture_model_sha256"]
    wheel = inputs / config["inputs"]["wheel"]["uri"].rsplit("/", 1)[1]
    prefix = config["serving_prefix"].copy()
    prefix[prefix.index("--from") + 1] = "vllm @ " + wheel.as_uri() + "#sha256=" + config["inputs"]["wheel"]["sha256"]
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0",
        "UV_NO_CONFIG": "1",
        "UV_CONSTRAINT": "/app/experiments/evaluation/agentx/requirements-vllm-hero.txt",
        "VLLM_CACHE_ROOT": str(inputs / "cache/vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(inputs / "cache/inductor"),
        "TRITON_CACHE_DIR": str(inputs / "cache/triton"),
    }
    outputs = []
    for index, mode in enumerate(config["runs"]):
        output = args.output / f"{index}-{mode}.json"
        command = [
            *prefix,
            *config["cuda_bootstrap"],
            "python",
            str(Path(__file__).with_name("tiny_graph_probe.py")),
            "--model",
            str(fixture),
            "--mode",
            mode,
            "--capture-sizes",
            *map(str, config["capture_sizes"]),
            "--output",
            str(output),
        ]
        (args.output / f"{index}-command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (args.output / f"{index}-{mode}.log").open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
        outputs.append(json.loads(output.read_text()))
        print("TINY_REPEAT_COMPLETED", index, mode, flush=True)
    comparisons = []
    for left, right in ((0, 1), (2, 3), (0, 2), (1, 3)):
        rows = []
        for a, b in zip(outputs[left]["rows"], outputs[right]["rows"], strict=True):
            assert a["branches"] == b["branches"]
            requests = []
            for at, ap, bt, bp in zip(a["tokens"], a["logprobs"], b["tokens"], b["logprobs"], strict=True):
                requests.append(
                    compare(
                        {"logprobs": {"tokens": at, "top_logprobs": ap}},
                        {"logprobs": {"tokens": bt, "top_logprobs": bp}},
                    )
                )
            rows.append({"branches": a["branches"], "requests": requests})
        comparisons.append({"left": left, "right": right, "rows": rows})
    (args.output / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")


if __name__ == "__main__":
    main()
