"""The actual trainer teardown preserves training failures and drains checkpoints."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from skyrl_train.evidence import TrainingEvidence
from skyrl_train.rollouts.buffer import Admission
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.telemetry import ProcessTelemetry, TRAINER_ROLE, TelemetryConfig
from skyrl_train.trainer import RayPPOTrainer


class TrainingFailed(Exception):
    pass


class CheckpointFailed(Exception):
    pass


class ClosingContext(TrainingContext):
    """Only the remote actor boundary is needed by the real close implementation."""

    def __init__(self, finish):
        self._dispatcher = None
        self._running = set()
        self._buffer = SimpleNamespace(finish=SimpleNamespace(remote=finish))
        self._policy_step = 0


class LifecycleTrainer(RayPPOTrainer):
    """Exercise real train/shutdown without constructing model actors."""

    def __init__(self, context, events, training_error):
        self.context = context
        self.events = events
        self.training_error = training_error
        self._rollout_spans_enabled = False
        self._distillation_runtime = None
        self.trajectory_runner = self.eval_trajectory_runner = object()
        self.global_step = 1

    async def _startup_trajectory_runner(self):
        self.events.append("startup")

    async def _train_loop(self):
        self.events.append("train")
        if self.training_error is not None:
            raise self.training_error

    async def _drain_checkpoint_upload(self):
        self.events.append("checkpoint_drain")
        raise CheckpointFailed("checkpoint upload unavailable")

    async def _teardown(self):
        self.events.append("teardown")


@pytest.mark.parametrize(
    "training_failed,closure",
    [(True, "failed"), (False, "failed"), (False, "hung"), (False, "successful")],
)
def test_actual_trainer_preserves_primary_and_drains_checkpoints_after_final_accounting_failure(
    tmp_path,
    monkeypatch,
    training_failed,
    closure,
):
    events = []
    primary = TrainingFailed("original training failure") if training_failed else None

    async def finish():
        events.append("finish")
        if closure == "hung":
            await asyncio.Event().wait()
        if closure == "failed":
            raise OSError("actor final accounting unavailable")
        return Admission(0, [], [], [], [], 0, None), 0

    monkeypatch.setattr("skyrl_train.rollouts.context._FINAL_ACCOUNTING_TIMEOUT", 0.01)
    monkeypatch.setattr("skyrl_train.rollouts.context.ray.kill", lambda actor: events.append("actor_killed"))
    monkeypatch.setattr("skyrl_train.trainer.write_exception_receipt", lambda *args: None)
    monkeypatch.setattr("skyrl_train.trainer.log_exception_as_text", lambda *args: None)
    context = ClosingContext(finish)
    trainer = LifecycleTrainer(context, events, primary)
    expected = (
        TrainingFailed
        if training_failed
        else {
            "failed": OSError,
            "hung": TimeoutError,
            "successful": CheckpointFailed,
        }[closure]
    )
    with pytest.raises(expected) as caught:
        with ProcessTelemetry(
            TelemetryConfig(),
            TRAINER_ROLE,
            TrainingEvidence(str(tmp_path), expected_steps=1, declarations={}),
        ):
            asyncio.run(trainer.train())
    if training_failed:
        assert caught.value is primary
    assert events == ["startup", "train", "finish", "actor_killed", "checkpoint_drain", "teardown"]
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["complete"] is False and manifest["status"] == "failed"
    assert manifest["closed_boundary"]["accounting_complete"] is (closure == "successful")
    rows = [
        json.loads(line)
        for chunk in manifest["chunks"]
        for line in (tmp_path / chunk["path"]).read_bytes().splitlines()
    ]
    failures = {row["attributes"]["phase"] for row in rows if row["name"] == "shutdown_failure"}
    assert "checkpoint_drain" in failures
    assert ("rollout_close" in failures) is (closure != "successful")
