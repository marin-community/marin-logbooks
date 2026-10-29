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
import urllib.request
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
    if manifest.get("diagnostic_request", {}).get("trace_batches"):
        probe.append("--trace-batches")
    (args.output / "cache-probe-command.json").write_text(json.dumps(probe, indent=2) + "\n")
    run_logged(probe, dict(os.environ), args.output / "cache-probe.log", timeout=900)
    comparison = json.loads((args.output / "cache-probe-matched" / "comparison.json").read_text())
    assert all(
        row["same_tokens"] and row["finite_logprobs"] and row["max_shared_logprob_absolute_delta"] == 0
        for row in comparison["comparisons"]
    )
    if manifest.get("diagnostic_request", {}).get("numerics_only"):
        print("NATIVE_NUMERICAL_CONTROL_DONE", flush=True)
        return
    if manifest.get("diagnostic_request", {}).get("session_affinity"):
        metrics_url = "http://127.0.0.1:38197/metrics"
        with urllib.request.urlopen(metrics_url, timeout=30) as response:
            (args.output / "affinity-metrics-before.txt").write_bytes(response.read())
        session_output = args.output / "cache-probe-session"
        session_probe = probe.copy()
        session_probe[session_probe.index("--output") + 1] = str(session_output)
        # CRC32 maps this stable ASCII ID to rank zero for sixteen engines.
        session_probe += ["--session-id", "hero-agentx-affinity-12"]
        run_logged(session_probe, dict(os.environ), args.output / "session-probe.log", timeout=900)
        with urllib.request.urlopen(metrics_url, timeout=30) as response:
            (args.output / "affinity-metrics-after.txt").write_bytes(response.read())
        session_comparison = json.loads((session_output / "comparison.json").read_text())
        assert all(
            row["same_tokens"] and row["finite_logprobs"] and row["max_shared_logprob_absolute_delta"] == 0
            for row in session_comparison["comparisons"]
        )
        explicit_rows = (args.output / "cache-probe-matched" / "requests.jsonl").read_text().splitlines()
        session_rows = (session_output / "requests.jsonl").read_text().splitlines()
        parity = []
        for explicit_text, session_text in zip(explicit_rows, session_rows, strict=True):
            explicit, session = json.loads(explicit_text), json.loads(session_text)
            assert explicit["wave"] == session["wave"] and explicit["branches"] == session["branches"]
            parity.append(
                {
                    "wave": explicit["wave"],
                    "exact_choices": explicit["response"]["choices"] == session["response"]["choices"],
                }
            )
        (args.output / "affinity-numerics.json").write_text(json.dumps(parity, indent=2) + "\n")
        assert all(row["exact_choices"] for row in parity)
    cases = json.loads(Path(__file__).with_name("experiments").joinpath("native-diagnostics.json").read_text())["cases"]
    decode = next(
        case
        for case in cases
        if "--random-output-len" in case["argv"] and case["argv"][case["argv"].index("--random-output-len") + 1] == "64"
    )
    prefill = next(case for case in cases if case["name"] == "long64k-c16")
    selected = [("decode-first", decode), ("decode-repeat", decode), ("long64k-c16", prefill)]
    selected += [(case["name"], case) for case in manifest.get("diagnostic_request", {}).get("additional_cases", [])]
    for name, case in selected:
        output = args.output / "clients" / name
        output.mkdir(parents=True, exist_ok=False)
        argv = [item for item in case["argv"] if item != "--profile"]
        argv += ["--result-dir", str(output), "--result-filename", "result.json"]
        command = prefix + argv
        record = {"source": manifest["source"], "command": command, "started_unix": time.time(), "diagnostic": True}
        (output / "command.json").write_text(json.dumps(record, indent=2) + "\n")
        # Native counters bracket the entire client process, outside its timed requests.
        with urllib.request.urlopen("http://127.0.0.1:38197/metrics", timeout=30) as response:
            (output / "server-metrics-before.txt").write_bytes(response.read())
        run_logged(command, dict(os.environ), output / "client.log", timeout=1800)
        with urllib.request.urlopen("http://127.0.0.1:38197/metrics", timeout=30) as response:
            (output / "server-metrics-after.txt").write_bytes(response.read())
        result = json.loads((output / "result.json").read_text())
        assert result["failed"] == 0 and result["completed"] == int(argv[argv.index("--num-prompts") + 1])
        record["finished_unix"] = time.time()
        (output / "command.json").write_text(json.dumps(record, indent=2) + "\n")
        print("NATIVE_CONTROL_DONE", name, flush=True)


if __name__ == "__main__":
    main()
