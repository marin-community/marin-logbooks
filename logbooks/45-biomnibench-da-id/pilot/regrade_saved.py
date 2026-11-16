# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry one failed grade from sealed reports using the original MiniMax settings."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from iris.rpc import job_pb2
from llm_judge import JudgeConfig, evaluate, parse_rubric, score_response
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from openai import OpenAI
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

ROOT = Path(__file__).resolve().parent
HASHES = {
    "answer.txt": "e2002dd6f5ea12d7da0078aab015541fed365881faef144834357b7d206e21db",
    "trace.md": "edb69b94506f84b3d53a54024bc005adf79838ba8f924b8c2430a18ae00a1300",
    "judge-response.json": "c15f5561a772dd11449c232161d231970687517e01095305377eadf9b80f6856",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output_prefix.rstrip("/") == args.source_prefix.rstrip("/"):
        raise ValueError("Recovery must have a separate output prefix")
    configure_coreweave_s3()
    destination = StoragePath(args.output_prefix)
    if (destination / "provenance.json").exists():
        raise ValueError("Recovery prefix already used; inspect before another attempt")
    rubric_path = ROOT / "remaining-tasks/da-20-3/tests/rubric.txt"
    levels = parse_rubric(rubric_path.read_text())
    with TemporaryDirectory(prefix="biomnibench-regrade-") as temporary:
        work = Path(temporary)
        submission = work / "submission"
        output = work / "grade"
        submission.mkdir()
        for name, expected in HASHES.items():
            data = (StoragePath(args.source_prefix) / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source checksum mismatch: {name}")
            (submission / name).write_bytes(data)
        original = json.loads((submission / "judge-response.json").read_text())
        assert original["choices"][0]["message"]["content"] == "{}"
        config = JudgeConfig(**json.loads((ROOT / "remaining-tasks/da-20-3/tests/judge-config.json").read_text()))
        provenance = {
            "task": "da-20-3",
            "source_prefix": args.source_prefix,
            "source_commit": args.source_commit,
            "input_sha256": HASHES,
            "rubric_sha256": hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
            "judge_config_sha256": hashlib.sha256((ROOT / "judge.yaml").read_bytes()).hexdigest(),
            "scope": "one judge-only retry; original reports, rubric, prompt and settings",
            "request_limit": 1,
            "solver_calls": 0,
            "original_result": "grader_error: empty JSON object",
        }
        if args.validate_only:
            print(json.dumps(provenance, indent=2))
            return
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2))
        judge = load_model_config(ROOT / "judge.yaml")
        inference = inference_config_for_model(
            judge,
            AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
            env_vars={},
            priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
            api_model=judge.name,
        )
        try:
            with remote_inference(inference) as session:
                endpoint = session.model.endpoint
                config = replace(config, model=endpoint.model, base_url=endpoint.base_url)
                with OpenAI(
                    api_key=endpoint.api_key or "unused",
                    base_url=endpoint.base_url,
                    timeout=config.timeout_sec,
                    max_retries=0,
                ) as client:
                    evaluate(rubric_path, submission, output, config, client)
                response = json.loads((output / "judge-response.json").read_text())
                grade = score_response(response["choices"][0]["message"]["content"], levels)
                print(json.dumps({"status": "graded", "score": grade["total_score"]}))
        finally:
            # Preserve failed replies as well as grades; never replace the native record.
            for path in sorted(output.glob("*")):
                (destination / path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
