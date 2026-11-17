# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade the four frozen submissions independently of Daytona capacity."""

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
        recover_grades(session, "s3://marin-us-east-02a/marin/evals/biomnibench-four-frozen-" + os.environ["MARIN_GIT_SHA"])


if __name__ == "__main__":
    main()
