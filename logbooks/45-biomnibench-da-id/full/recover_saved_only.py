# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade the four frozen submissions independently of Daytona capacity."""

import argparse
import os
from pathlib import Path

from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from recovery_executor import recover_grades
from rigging.filesystem.s3_compat import configure_coreweave_s3

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-grade", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    judge = load_model_config(ROOT.parent / "pilot/judge.yaml")
    config = inference_config_for_model(
        judge,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        env_vars={},
        priority=job_pb2.PRIORITY_BAND_INHERIT,
        api_model=judge.name,
    )
    with remote_inference(config) as session:
        campaign = "biomnibench-final-da-13-5" if args.final_grade else "biomnibench-four-frozen"
        recover_grades(
            session,
            f"s3://marin-us-east-02a/marin/evals/{campaign}-" + os.environ["MARIN_GIT_SHA"],
            selected_task="da-13-5" if args.final_grade else None,
        )


if __name__ == "__main__":
    main()
