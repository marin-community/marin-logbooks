"""Retained producer evidence remains usable when the live telemetry sink is absent."""

import asyncio
import hashlib
import json
import threading
from pathlib import Path

import pytest

from skyrl_train.evidence import TrainingEvidence
from skyrl_train.io import io
from skyrl_train.rollout_observability import (
    async_phase_window,
    record_group_disposition,
)
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    GroupRewards,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutVerdict,
)
from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.telemetry import (
    GeneratedWork,
    ProcessTelemetry,
    TRAINER_ROLE,
    TelemetryConfig,
    record_event,
    record_generated_work,
    record_training_metrics,
)


def retained_records(root: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((root / "manifest.json").read_bytes())
    rows = []
    for chunk in manifest["chunks"]:
        payload = (root / chunk["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == chunk["sha256"]
        rows.extend(json.loads(line) for line in payload.splitlines())
    assert [row["sequence"] for row in rows] == list(range(manifest["records"]))
    return manifest, rows


@pytest.mark.parametrize("second_applied", [True, False])
def test_training_evidence_retains_applied_skipped_age_and_terminal_flush_without_live_sink(
    tmp_path,
    second_applied,
):
    sink = TrainingEvidence(str(tmp_path), expected_steps=2, declarations={"max_staleness_steps": 2})
    with ProcessTelemetry(TelemetryConfig(), TRAINER_ROLE, sink):
        for step, applied, skipped in [(1, 1, 0), (2, int(second_applied), int(not second_applied))]:
            record_generated_work(GeneratedWork(1, 1, 7), step, uid=f"group-{step}")
            record_event(
                "consumed_staleness",
                {"staleness": 1, "sequences": 1, "response_tokens": 7},
                attributes={"step": str(step), "uid": f"group-{step}"},
            )
            record_event(
                "selected_token_staleness",
                {
                    "async/staleness_tokens_count": 7,
                    "async/staleness_tokens_mean": 1.0,
                    "async/staleness_tokens_max": 1.0,
                },
                attributes={"step": str(step)},
            )
            with async_phase_window("training", step=step, enabled=True):
                record_event(
                    "optimizer_window",
                    {"applied": bool(applied), "raw_grad_norm": 1.0, "policy_loss": 0.2, "loss_finite": True},
                    attributes={"step": str(step), "window": "0"},
                )
                record_event(
                    "policy_update",
                    {
                        "policy_update_steps": applied,
                        "skipped_steps": skipped,
                        "raw_grad_norm": 1.0,
                        "policy_loss": 0.2,
                    },
                    attributes={"step": str(step)},
                )
            record_training_metrics({"policy/policy_update_steps": applied}, step=step, kind="train")
            record_group_disposition(
                uid=f"group-{step}",
                disposition="consumed",
                tokens=7,
                step=step,
                dwell_seconds=0.0,
            )
        record_event(
            "rollout_closed",
            {
                "driver_dispatch_stopped": True,
                "accounting_complete": True,
                "uncommitted_leases": 1,
                "uncommitted_tokens_known": False,
            },
        )
    manifest, rows = retained_records(tmp_path)
    assert manifest["complete"] is True
    assert manifest["all_updates_finite_and_applied"] is second_applied
    assert manifest["optimizer_window_counts_match"] is True
    assert manifest["policy_updates"]["2"]["metrics"]["skipped_steps"] == int(not second_applied)
    assert manifest["selected_group_response_tokens_by_step_and_group_age"] == {
        "1": {"1": 7},
        "2": {"1": 7},
    }
    assert manifest["selected_loss_token_age_metrics_by_outer_step"]["1"]["async/staleness_tokens_count"] == 7
    assert manifest["disposition_tokens"] == {"consumed": 14}
    assert manifest["closed_boundary"]["uncommitted_leases"] == 1
    assert manifest["uncommitted_generation_tokens"] is None
    assert sum(row["name"] == "training_metrics" for row in rows) == 2
    assert sum(row["name"] == "async_phase_window" for row in rows) == 2
    assert {row["attributes"]["uid"] for row in rows if row["name"] == "consumed_staleness"} == {"group-1", "group-2"}
    assert all((tmp_path / chunk["path"]).stat().st_mode & 0o777 == 0o600 for chunk in manifest["chunks"])


def test_training_evidence_interrupt_preserves_original_exception_and_incomplete_history(
    tmp_path,
):
    class TrainingFailed(Exception):
        pass

    with pytest.raises(TrainingFailed):
        with ProcessTelemetry(
            TelemetryConfig(),
            TRAINER_ROLE,
            TrainingEvidence(str(tmp_path), expected_steps=2, declarations={}),
        ):
            record_event(
                "consumed_staleness",
                {"staleness": 0, "sequences": 1, "response_tokens": 3},
                attributes={"step": "1", "uid": "first"},
            )
            raise TrainingFailed("original training failure")
    manifest, rows = retained_records(tmp_path)
    assert manifest["status"] == "failed" and manifest["complete"] is False
    assert manifest["missing_outer_step_count"] == 2
    assert len(rows) == 1


def test_training_evidence_failed_chunk_write_cannot_publish_complete(tmp_path, monkeypatch):
    original = io.write_bytes_atomic

    def fail_chunk(path, payload):
        if "/chunks/" in path:
            raise OSError("synthetic unavailable object store")
        original(path, payload)

    monkeypatch.setattr(io, "write_bytes_atomic", fail_chunk)
    with pytest.raises(OSError):
        with ProcessTelemetry(
            TelemetryConfig(),
            TRAINER_ROLE,
            TrainingEvidence(str(tmp_path), expected_steps=1, declarations={}),
        ):
            record_event(
                "policy_update",
                {
                    "policy_update_steps": 1,
                    "skipped_steps": 0,
                    "raw_grad_norm": 1.0,
                    "policy_loss": 0.0,
                },
                attributes={"step": "1"},
            )
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["status"] == "recording" and manifest["complete"] is False
    assert manifest["chunks"] == []


def test_training_evidence_nonfinite_update_is_explicit_not_silently_dropped(tmp_path):
    with ProcessTelemetry(
        TelemetryConfig(),
        TRAINER_ROLE,
        TrainingEvidence(str(tmp_path), expected_steps=1, declarations={}),
    ):
        record_event(
            "consumed_staleness",
            {"staleness": 0, "sequences": 1, "response_tokens": 3},
            attributes={"step": "1", "uid": "first"},
        )
        record_event(
            "policy_update",
            {
                "policy_update_steps": 0,
                "skipped_steps": 1,
                "raw_grad_norm": float("nan"),
                "policy_loss": 0.2,
            },
            attributes={"step": "1"},
        )
        record_event(
            "rollout_closed", {"driver_dispatch_stopped": True, "accounting_complete": True, "uncommitted_leases": 0}
        )
    manifest, rows = retained_records(tmp_path)
    assert manifest["all_updates_finite_and_applied"] is False
    update = next(row for row in rows if row["name"] == "policy_update")
    assert update["fields"]["raw_grad_norm"] is None and update["nonfinite_fields"] == ["raw_grad_norm"]


def test_buffer_terminal_drain_keeps_unreported_generation_and_untrained_disposition():
    async def finish_run():
        buffer = RolloutBuffer(RolloutBufferConfig(1, 3, 2, BatchPolicy.ROLLING, None, None))
        await buffer.publish(1)
        first = await buffer.acquire_lease()
        verdict = RolloutVerdict(
            "first",
            (),
            GroupSelectionResult.KEEP,
            GroupRewards((1.0,), (1.0,), True),
            GeneratedWork(1, 1, 3),
        )
        await buffer.commit(first.lease_id, {"uid": "first"}, verdict, PayloadReference("first-payload"))
        consumed = await buffer.admit(1.0)
        second = await buffer.acquire_lease()
        await buffer.acquire_lease()
        verdict = RolloutVerdict(
            "second",
            (),
            GroupSelectionResult.KEEP,
            GroupRewards((1.0,), (1.0,), True),
            GeneratedWork(1, 1, 5),
        )
        await buffer.commit(
            second.lease_id,
            {"uid": "second"},
            verdict,
            PayloadReference("second-payload"),
        )
        terminal, leases = await buffer.finish()
        return consumed, terminal, leases

    consumed, terminal, leases = asyncio.run(finish_run())
    assert [(row.uid, row.disposition) for row in consumed.dispositions] == [("first", "consumed")]
    assert [(row.uid, row.disposition, row.tokens) for row in terminal.dispositions] == [
        ("second", "terminal_untrained", 5)
    ]
    assert [(uid, work.generated_token_count) for uid, _, work in terminal.generated] == [("second", 5)]
    assert leases == 1


def test_blocked_publication_keeps_thread_and_async_producers_responsive_and_exactly_once(tmp_path, monkeypatch):
    original = io.write_bytes_atomic
    entered, release = threading.Event(), threading.Event()
    paths = []

    def blocked_write(path, payload):
        if "/chunks/" in path:
            paths.append(path)
            if not entered.is_set():
                entered.set()
                assert release.wait(5), "test did not release storage boundary"
        original(path, payload)

    monkeypatch.setattr(io, "write_bytes_atomic", blocked_write)
    sink = TrainingEvidence(str(tmp_path), expected_steps=20, declarations={})

    def policy_updates():
        for step in range(1, 21):
            record_event("policy_update", {"policy_update_steps": 1}, attributes={"step": str(step)})

    async def rollouts():
        for call in range(80):
            record_event("rollout_call", {"call": call})
            if call % 5 == 0:
                sink.flush()
            await asyncio.sleep(0)

    with ProcessTelemetry(TelemetryConfig(), TRAINER_ROLE, sink):
        producer = threading.Thread(target=policy_updates)
        try:
            producer.start()
            assert entered.wait(2)
            asyncio.run(rollouts())
            producer.join(2)
            assert not producer.is_alive()
            assert not release.is_set()  # Both producers finished while cloud I/O was still blocked.
        finally:
            release.set()
            producer.join(2)
    manifest, rows = retained_records(tmp_path)
    assert len(rows) == manifest["records"] == 100
    assert {row["fields"]["call"] for row in rows if row["name"] == "rollout_call"} == set(range(80))
    assert {row["attributes"]["step"] for row in rows if row["name"] == "policy_update"} == {
        str(i) for i in range(1, 21)
    }
    assert len(paths) == len(set(paths)) == len(manifest["chunks"])
    assert manifest["bytes"] == sum(chunk["bytes"] for chunk in manifest["chunks"])
    assert manifest["unacknowledged_bytes"] == 0 and manifest["rejected_records"] == 0


def test_publisher_pressure_is_bounded_explicit_and_retains_all_accepted_records(tmp_path, monkeypatch):
    original = io.write_bytes_atomic
    entered, release = threading.Event(), threading.Event()

    def blocked_write(path, payload):
        if "/chunks/" in path and not entered.is_set():
            entered.set()
            assert release.wait(5)
        original(path, payload)

    monkeypatch.setattr(io, "write_bytes_atomic", blocked_write)
    sink = TrainingEvidence(str(tmp_path), expected_steps=0, declarations={})
    sink.start()
    try:
        sink.record("rollout_call", {"call": 0}, {})
        sink.flush()
        assert entered.wait(2)
        for call in range(1, 81):
            sink.record("rollout_call", {"call": call, "detail": "x" * 30000}, {})
            sink.flush()
    finally:
        release.set()
    with pytest.raises(BufferError, match="queue exhausted"):
        sink.close(failed=False)
    manifest, rows = retained_records(tmp_path)
    assert manifest["status"] == "failed" and manifest["complete"] is False
    assert manifest["failure_class"] == "BufferError" and manifest["rejected_records"] > 0
    assert len(rows) + manifest["rejected_records"] == 81
    assert manifest["unacknowledged_bytes"] == 0


def test_close_deadline_does_not_wait_for_blocked_storage_or_claim_completion(tmp_path, monkeypatch):
    original = io.write_bytes_atomic
    entered, release, failed_published = threading.Event(), threading.Event(), threading.Event()

    def blocked_write(path, payload):
        if "/chunks/" in path:
            entered.set()
            assert release.wait(5)
        original(path, payload)
        if path.endswith("manifest.json") and json.loads(payload)["failure_class"] == "TimeoutError":
            failed_published.set()

    monkeypatch.setattr(io, "write_bytes_atomic", blocked_write)
    sink = TrainingEvidence(str(tmp_path), expected_steps=0, declarations={}, close_timeout=0.01)
    sink.start()
    sink.record("rollout_call", {"call": 0}, {})
    sink.flush()
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="close deadline"):
            sink.close(failed=False)
        assert not release.is_set()
        assert json.loads((tmp_path / "manifest.json").read_bytes())["complete"] is False
    finally:
        release.set()
    assert failed_published.wait(2)
    manifest, rows = retained_records(tmp_path)
    assert manifest["status"] == "failed" and manifest["complete"] is False
    assert len(rows) == 1


def test_late_final_manifest_ack_is_followed_by_failed_projection_after_close_deadline(tmp_path, monkeypatch):
    original = io.write_bytes_atomic
    entered, release, failed_published = threading.Event(), threading.Event(), threading.Event()

    def blocked_write(path, payload):
        if path.endswith("manifest.json"):
            manifest = json.loads(payload)
            if manifest["complete"]:
                entered.set()
                assert release.wait(5)
            original(path, payload)
            if manifest["failure_class"] == "TimeoutError":
                failed_published.set()
        else:
            original(path, payload)

    monkeypatch.setattr(io, "write_bytes_atomic", blocked_write)
    sink = TrainingEvidence(str(tmp_path), expected_steps=1, declarations={}, close_timeout=1.0)
    sink.start()
    sink.record("consumed_staleness", {"sequences": 1, "staleness": 0, "response_tokens": 3}, {"step": "1"})
    sink.record("optimizer_window", {"applied": True, "raw_grad_norm": 1.0, "loss_finite": True}, {"step": "1"})
    sink.record(
        "policy_update",
        {"policy_update_steps": 1, "skipped_steps": 0, "raw_grad_norm": 1.0, "policy_loss": 0.2},
        {"step": "1"},
    )
    sink.record("rollout_closed", {"driver_dispatch_stopped": True, "accounting_complete": True}, {})
    try:
        with pytest.raises(TimeoutError):
            sink.close(failed=False)
        assert entered.is_set() and not release.is_set()
    finally:
        release.set()
    assert failed_published.wait(2)
    manifest, rows = retained_records(tmp_path)
    assert manifest["status"] == "failed" and manifest["complete"] is False
    assert len(rows) == 4
