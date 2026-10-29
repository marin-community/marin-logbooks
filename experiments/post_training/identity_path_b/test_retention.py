# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from queue import Queue

import fsspec
import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_path_b import export_preflight, retention, workflow
from experiments.post_training.identity_path_b.async_launch import ExportBinding, TrendInputs
from experiments.post_training.identity_path_b.test_workflow import sample_event


def test_two_queued_or_failed_exports_hold_slots_nonfinal_skips_final_waits(monkeypatch, tmp_path):
    paths = [tmp_path / "step-1", tmp_path / "step-2"]
    for p in paths:
        p.mkdir()
    monkeypatch.setattr(retention, "StoragePath", lambda p: p)
    monkeypatch.setattr(retention.time, "monotonic", lambda: 0)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        monkeypatch.setattr(retention.time, "monotonic", lambda: 7201)

    monkeypatch.setattr(retention.time, "sleep", sleep)
    assert not retention.await_checkpoint_slot(paths, final=False)
    assert all(p.exists() for p in paths) and sleeps == [2]

    # No final timeout even with clock beyond the nonfinal deadline.
    def release(seconds):
        sleeps.append(seconds)
        paths[0].rmdir()

    monkeypatch.setattr(retention.time, "sleep", release)
    assert retention.await_checkpoint_slot(paths, final=True)
    assert sleeps == [2, 2] and paths[1].exists()


def test_probe_failure_prevents_any_gpu_submission(monkeypatch):
    event = sample_event()
    calls = []

    def fail(event):
        raise PermissionError("write suspended")

    monkeypatch.setattr(workflow, "export_write_probe", fail)
    monkeypatch.setattr(workflow, "native_submit", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(workflow, "write_json", lambda *a: None)
    binding = ExportBinding("test", b"{}", "a" * 64, {}, {}, TrendInputs("id", "ctl"))
    owner = workflow.Workflow("unused", binding, Queue())
    owner.checkpoint(event)
    assert not calls and not owner.active
    assert owner.records["checkpoint-000025"]["status"] == "export_preflight_failed_no_gpu"


def test_tiny_probe_retains_exact_payload_without_overwrite(monkeypatch, tmp_path):
    fs = fsspec.filesystem("memory")
    fs.store.clear()

    def local(path):
        return StoragePath("memory://" + path.removeprefix("s3://"))

    monkeypatch.setattr(export_preflight, "StoragePath", local)
    event = sample_event()
    proof = export_preflight.export_write_probe(event)
    assert proof["bytes"] == 32 and proof["application_attempts"] == 1
    with pytest.raises(FileExistsError):
        export_preflight.export_write_probe(event)


def test_only_exact_audited_current_r5_temporary_native_is_deleted(monkeypatch, tmp_path):
    fs = fsspec.filesystem("memory")
    fs.store.clear()

    def local(path):
        return StoragePath("memory://" + path.removeprefix("s3://"))

    monkeypatch.setattr(retention, "StoragePath", local)
    event = sample_event()
    root = "s3://marin-rno2a/tmp/ttl=14d/path-b/r5-fixture/arm10/run"
    event = replace(event, native_checkpoint=root + "/checkpoints/step-24", hf_export=root + "/hf/updates-25")
    checkpoint = local(event.native_checkpoint)
    checkpoint.mkdirs()
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 24, "is_temporary": True}))
    local(event.hf_export + "/path-b-checkpoint-event.json").write_text(json.dumps(event.payload()))
    audit = {
        "schema": "path_b_storage_v1",
        "source_receipt_sha256": "SOURCE",
        "frozen_native_tensors": 104,
        "trained_master_cast_tensors": 398,
        "tensors": {str(i): {"frozen_source": i < 104} for i in range(502)},
    }
    audit_path = local(event.hf_export + "/path-b-storage-audit.json")
    audit_path.write_text(json.dumps({**audit, "trained_master_cast_tensors": 397}))
    with pytest.raises(ValueError, match="full502"):
        retention.delete_audited_checkpoint(event, root, "SOURCE")
    assert checkpoint.exists()
    audit_path.write_text(json.dumps(audit))
    retention.delete_audited_checkpoint(event, root, "SOURCE")
    assert not checkpoint.exists()
    assert local(root + "/checkpoint-retirement/update-25.json").exists()


def test_real_native_out_of_order_retirement_never_prunes_export_in_use(tmp_path):
    import jax.numpy as jnp  # noqa: PLC0415
    from levanter.checkpoint import CheckpointerConfig, CheckpointRetention  # noqa: PLC0415

    config = CheckpointerConfig(
        base_path=str(tmp_path),
        append_run_id_to_base_path=False,
        save_interval=None,
        keep=None,
        keep_last_temporary_checkpoints=2,
        delete_old_temp_checkpoints=False,
    )
    writer = config.create("fixture")
    tree = {"value": jnp.arange(8)}
    # Reproduce real native temp tracking A/B before B retires out of order.
    for step in (1, 2):
        writer.request_checkpoint(CheckpointRetention.TEMPORARY)
        writer.on_step(tree=tree, step=step, force=False)
        writer.wait_until_finished()
    assert len(writer._temporary_checkpoints) == 2
    StoragePath(str(tmp_path / "step-2")).rmtree()
    assert retention.await_checkpoint_slot([str(tmp_path / "step-1"), str(tmp_path / "step-2")], final=False)
    retention.save_temporary_checkpoint(writer, tree, step=3)
    writer.wait_until_finished()
    assert (tmp_path / "step-1" / "metadata.json").exists()
    assert not (tmp_path / "step-2").exists()
    assert json.loads((tmp_path / "step-3" / "metadata.json").read_text())["is_temporary"] is True
    assert len(list(tmp_path.glob("step-*"))) == 2
