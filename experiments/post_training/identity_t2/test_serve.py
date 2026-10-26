# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.types import Entrypoint, EnvironmentSpec

from experiments.post_training.identity_t2.serve import environment_setup_scripts, setup_script, worker_command


def test_native_compiled_setup_keeps_root_provisioning_before_the_separate_profile():
    native_root = list(EnvironmentSpec(extras=["gpu"]).to_proto().setup_scripts)
    config = EnvironmentSpec(extras=["gpu"], setup_scripts=environment_setup_scripts()).to_proto()
    runtime = build_runtime_entrypoint(Entrypoint(command=worker_command()), config)
    assert list(runtime.setup_commands[: len(native_root)]) == native_root
    assert runtime.setup_commands[len(native_root)] == setup_script()
    # Iris, not this experiment, installs runtime dependencies once at the end.
    assert len(runtime.setup_commands) == len(native_root) + 2
    subprocess.run(["bash", "-n"], input="\n".join(runtime.setup_commands), text=True, check=True)


def test_worker_executes_root_python_while_profile_vllm_remains_first_on_path(tmp_path):
    root = tmp_path / "root"
    profile = tmp_path / "profile"
    for directory in (root / "bin", profile / "bin"):
        directory.mkdir(parents=True)
    root_python = root / "bin/python"
    root_python.write_text('#!/bin/bash\nprintf "ROOT_WORKER\\n"\ncommand -v vllm\nprintf "%s\\n" "$*"\n')
    profile_python = profile / "bin/python"
    profile_python.write_text("#!/bin/bash\necho WRONG_PROFILE_WORKER\nexit 7\n")
    vllm = profile / "bin/vllm"
    vllm.write_text("#!/bin/bash\nexit 0\n")
    for executable in (root_python, profile_python, vllm):
        executable.chmod(0o700)
    activation = tmp_path / "profile-activation.sh"
    activation.write_text(f'export PATH="{profile / "bin"}:$PATH"\n')
    # External executables are stubs; exercise the actual shell command and PATH
    # transition without provisioning dependencies or loading a model.
    env = {**os.environ, "IRIS_VENV": str(root), "IRIS_PYTHON": "python"}
    result = subprocess.run(worker_command(activation), env=env, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert lines == ["ROOT_WORKER", str(vllm), "-u -m experiments.post_training.identity_t2.serve worker"]
