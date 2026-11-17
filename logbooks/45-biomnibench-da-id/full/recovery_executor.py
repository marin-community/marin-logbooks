# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover frozen grades and prepare the selected tasks while serving is warm."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from marin.evaluation.runner import EvalExecutor, EvaluationOutcome
from marin.inference.iris import RemoteInferenceSession
from marin.inference.types import InferenceBackendState
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL, DaytonaConnection
from shellbox.backends.daytona.snapshot import SnapshotRequest, ensure_snapshot

ROOT = Path(__file__).resolve().parent


async def prepare_snapshots(api_key: str) -> None:
    connection = DaytonaConnection(api_key, DEFAULT_DAYTONA_API_URL, "us")
    specs = json.loads((ROOT / "recovery-snapshots.json").read_text())
    async with connection.client_factory()() as client:
        for spec in specs:
            await ensure_snapshot(client, SnapshotRequest.from_dict(spec))
            print(json.dumps({"snapshot_ready": spec["name"]}), flush=True)


def recover_grades(judge: RemoteInferenceSession, output_dir: str) -> None:
    manifest = json.loads((ROOT / "recovery-inputs.json").read_text())
    endpoint = judge.model.endpoint
    destination = StoragePath(output_dir) / "judge-only-recovery"
    with TemporaryDirectory(prefix="biomnibench-recovery-") as temporary:
        for task, files in manifest["tasks"].items():
            if (destination / task / "recovery.json").exists():
                continue
            submission = Path(temporary) / task / "submission"
            output = submission.parent / "grade"
            submission.mkdir(parents=True)
            for name, expected in files.items():
                data = (StoragePath(manifest["prefix"]) / task / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Frozen input checksum mismatch: {task}/{name}")
                (submission / name).write_bytes(data)
            tests = ROOT / "tasks" / task / "tests"
            result = subprocess.run(
                [
                    sys.executable,
                    str(tests / "llm_judge.py"),
                    "--rubric",
                    str(tests / "rubric.txt"),
                    "--config",
                    str(tests / "judge-config.json"),
                    "--submission-dir",
                    str(submission),
                    "--output-dir",
                    str(output),
                ],
                env={
                    **os.environ,
                    "MODEL_NAME": endpoint.model,
                    "OPENAI_BASE_URL": endpoint.base_url,
                    "OPENAI_API_KEY": endpoint.api_key or "unused",
                },
                check=False,
                timeout=180,
            )
            for path in sorted(output.glob("*")):
                (destination / task / path.name).write_bytes(path.read_bytes())
            status = {
                "task": task,
                "exit_code": result.returncode,
                "input_sha256": files,
                "solver_calls": 0,
                "reconstructed_from_trajectory": task == "da-15-1",
            }
            (destination / task / "recovery.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)


@dataclass(frozen=True)
class RecoveryExecutor:
    delegate: EvalExecutor

    def __call__(
        self,
        session: RemoteInferenceSession,
        output_dir: str,
        env_vars: Mapping[str, str],
        *,
        backend_state: Callable[[], InferenceBackendState],
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome:
        assert judge is not None
        recover_grades(judge, output_dir)
        asyncio.run(prepare_snapshots(env_vars["DAYTONA_API_KEY"]))
        return self.delegate(session, output_dir, env_vars, backend_state=backend_state, judge=judge)
