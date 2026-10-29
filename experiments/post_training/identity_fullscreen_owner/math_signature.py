# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Call the real pinned f5 Evalchemy boundary without inference or child jobs."""

import argparse
import hashlib
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from experiments.post_training.identity_fullscreen_owner.owner import math_workspace

F5_RUNNER_SHA256 = "3dccdeb52c05573fe6f72b5d07f74cdf72a07b1aa52b06de370faa0370a0df70"


class ChildBoundaryReached(Exception):
    """Stop the real evaluator at its child-job boundary, before any RPC."""


def check_real_f5_signature():
    runner = importlib.import_module("marin.evaluation.evalchemy.runner")
    wrapper = importlib.import_module("experiments.post_training.identity_fullscreen_owner.frozen.run_math")
    if hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest() != F5_RUNNER_SHA256:
        raise ValueError("Signature smoke must execute the exact f5 runner source")
    if wrapper.native_run is not runner.run_evalchemy:
        raise ValueError("Wrapper did not retain the actual native f5 callable")
    if inspect.signature(runner.run_evalchemy).parameters["env_vars"].kind != inspect.Parameter.KEYWORD_ONLY:
        raise ValueError("Pinned f5 keyword-only boundary changed")
    env = {"SIGNATURE_FIXTURE": "not-a-secret"}
    output = "s3://signature-only.invalid/no-write"
    config = runner.EvalchemyRunConfig(name="signature-only", tasks=())
    try:
        wrapper.scoped_run(None, config, output, env, backend_state=lambda: None)
    except ValueError as error:
        if str(error) != "Evalchemy requires at least one task":
            raise
    else:
        raise AssertionError("Actual native no-task guard was not reached")

    def at_child(model, actual_config, actual_output, actual_env, *, backend_state):
        if actual_env is not env or actual_output != output or model is not None:
            raise AssertionError("Wrapper changed the native call arguments")
        if actual_config.runtime.dataset_cache != output + "/dataset-cache":
            raise AssertionError("Scoped cache path was lost")
        raise ChildBoundaryReached

    config = runner.EvalchemyRunConfig(name="signature-only", tasks=(object(),))
    with patch.object(runner, "_run_evalchemy_child", at_child):
        try:
            wrapper.scoped_run(None, config, output, env, backend_state=lambda: None)
        except ChildBoundaryReached:
            pass
        else:
            raise AssertionError("Actual native child boundary was not reached")
    return {"real_f5_signature": "PASS", "runner_sha256": F5_RUNNER_SHA256, "model_calls": 0, "child_submissions": 0}


def pinned_signature_smoke(archive: str, expected_sha: str):
    workspace = math_workspace(archive, expected_sha)
    try:
        env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONSAFEPATH="1")
        env["PYTHONPATH"] = ":".join(
            [str(workspace), str(workspace / "experiments/post_training/identity_fullscreen_owner/frozen")]
            + [str(p) for p in (workspace / "lib").glob("*/src")]
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-P",
                "-m",
                "experiments.post_training.identity_fullscreen_owner.math_signature",
                "--child",
            ],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(workspace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(check_real_f5_signature(), sort_keys=True))
