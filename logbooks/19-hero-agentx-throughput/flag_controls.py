# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed cache/state and native short-bank controls after server readiness.

These diagnostics select serving flags. They are not AgentX confirmation runs.
Use the server's exact isolated package prefix and the recorded tokenizer pins.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from experiments.evaluation.agentx.processes import run_logged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    server = json.loads((args.output / "diagnostic.json").read_text())
    assert server["status"] == "ready"
    command = server["server_command"]
    prefix = ["env", "UV_NO_CONFIG=1", "UV_CONSTRAINT=/app/experiments/evaluation/agentx/requirements-vllm-hero.txt"]
    prefix += command[: command.index("python")]
    assert "uvx" in prefix and "--from" in prefix
    tokenizer = Path("/tmp/hero-tokenizer")
    tokenizer.mkdir(exist_ok=True)
    lineage = manifest["workload"]["lineage"]
    for name, digest in lineage["tokenizer_files"].items():
        source = Path(hf_hub_download(lineage["tokenizer"], name, revision=lineage["tokenizer_revision"]))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        shutil.copyfile(source, tokenizer / name)
    probe = [
        *prefix,
        "python",
        str(Path(__file__).with_name("cache_probe.py")),
        "--tokenizer",
        str(tokenizer / "tokenizer.json"),
        "--output",
        str(args.output / "cache-probe-matched"),
    ]
    (args.output / "cache-probe-command.json").write_text(json.dumps(probe, indent=2) + "\n")
    run_logged(probe, dict(os.environ), args.output / "cache-probe.log", timeout=900)
    comparison = json.loads((args.output / "cache-probe-matched" / "comparison.json").read_text())
    assert all(
        row["same_tokens"] and row["finite_logprobs"] and row["max_shared_logprob_absolute_delta"] == 0
        for row in comparison["comparisons"]
    )
    cases = json.loads(Path(__file__).with_name("experiments").joinpath("native-diagnostics.json").read_text())["cases"]
    decode = next(
        case
        for case in cases
        if "--random-output-len" in case["argv"] and case["argv"][case["argv"].index("--random-output-len") + 1] == "64"
    )
    prefill = next(case for case in cases if case["name"] == "long64k-c16")
    for name, case in (("decode-first", decode), ("decode-repeat", decode), ("long64k-c16", prefill)):
        output = args.output / "clients" / name
        output.mkdir(parents=True, exist_ok=False)
        argv = [item for item in case["argv"] if item != "--profile"]
        argv += ["--result-dir", str(output), "--result-filename", "result.json"]
        command = prefix + argv
        record = {"source": manifest["source"], "command": command, "started_unix": time.time(), "diagnostic": True}
        (output / "command.json").write_text(json.dumps(record, indent=2) + "\n")
        run_logged(command, dict(os.environ), output / "client.log", timeout=1800)
        result = json.loads((output / "result.json").read_text())
        assert result["failed"] == 0 and result["completed"] == int(argv[argv.index("--num-prompts") + 1])
        record["finished_unix"] = time.time()
        (output / "command.json").write_text(json.dumps(record, indent=2) + "\n")
        print("NATIVE_CONTROL_DONE", name, flush=True)


if __name__ == "__main__":
    main()
