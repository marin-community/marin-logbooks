# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess

import pytest
import yaml
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import skyrl_cpu_command

from experiments.post_training.inventory import train_quick_loop


@pytest.mark.parametrize("steps", [3, 25])
def test_arms_change_only_declared_recipe_fields(steps, monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "research")
    selection = ArtifactStep.adopt("rl-data/task-selection/snowball-100k", "2026.10.23.2", "s3://test/selection")
    control = train_quick_loop.training_recipe(steps)
    inputs = []
    names = []
    executions = []
    for arm in train_quick_loop.Arm:
        expected = copy.deepcopy(control)
        if arm == train_quick_loop.Arm.LEARNING_RATE:
            expected["trainer"]["policy"]["optimizer_config"]["lr"] = 2e-6
        if arm == train_quick_loop.Arm.SEQUENCE_MEAN:
            expected["trainer"]["algorithm"]["loss_reduction"] = "sequence_mean"
        if arm == train_quick_loop.Arm.ASYNC:
            expected["trainer"]["rollout_buffer"].update(max_staleness_steps=2, max_in_flight=192)
        step = train_quick_loop.build_run("2026.10.24.1", selection=selection, steps=steps, arm=arm)
        launch = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        rendered = yaml.safe_load(launch.launch_config_yaml)
        assert rendered["skyrl"] == expected
        inputs.append(rendered["inputs"])
        executions.append(step.runtime_args)
        names.append(step.name)
    assert all(value == inputs[0] for value in inputs)
    assert all(value == executions[0] for value in executions)
    assert len(set(names)) == 4


@pytest.mark.timeout(180)  # The production bootstrap composes all eight plans in its frozen CPU environment.
def test_all_arms_compose_in_frozen_skyrl_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "research")
    selection = ArtifactStep.adopt("rl-data/task-selection/snowball-100k", "2026.10.23.2", "s3://test/selection")
    paths = []
    for steps in (3, 25):
        for arm in train_quick_loop.Arm:
            step = train_quick_loop.build_run("2026.10.24.1", selection=selection, steps=steps, arm=arm)
            launch = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
            path = tmp_path / f"{arm.value}-{steps}.yaml"
            path.write_text(launch.launch_config_yaml)
            paths.append(str(path))
    script = """
import json
import sys
from pathlib import Path
from cloud.iris.launch_config import load_launch_config
from skyrl_train import objective
from skyrl_train.utils import validate_cfg

results = []
for path in sys.argv[1:]:
    launch = load_launch_config(Path(path))
    validate_cfg(launch.skyrl)
    results.append({
        "entrypoint": launch.runtime.entrypoint,
        "training_type": launch.runtime.training_type,
        "steps": launch.skyrl.trainer.max_steps,
        "batch_policy": launch.skyrl.trainer.rollout_buffer.batch_policy,
        "evaluation": [launch.skyrl.trainer.eval_before_train, launch.skyrl.trainer.eval_interval],
    })
print(json.dumps(results))
"""
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    # The trainer validates W&B presence without connecting; isolate it from real credentials.
    environment["RIGGING_SECRETS_DIR"] = str(tmp_path / "synthetic-secrets")
    environment["WANDB_API_KEY"] = "offline-validation-only"
    result = subprocess.run(
        skyrl_cpu_command("python", "-c", script, *paths),
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {
            "entrypoint": "skyrl_train.entrypoints.taskcompendium",
            "training_type": "async" if arm == train_quick_loop.Arm.ASYNC else "sync",
            "steps": steps,
            "batch_policy": "full_batch",
            "evaluation": [True, steps],
        }
        for steps in (3, 25)
        for arm in train_quick_loop.Arm
    ]
