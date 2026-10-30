# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import threading
from concurrent.futures import Future

import pytest
from iris.rpc import job_pb2
from iris.testing.journeys.world import journey_world
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_path_b import (
    ablation_launch,
    ablation_recovery,
    async_launch,
    export_preflight,
    worker,
)
from experiments.post_training.identity_path_b.ablation import AblationArm, frozen_names
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.launch import SOURCE_RECEIPT
from experiments.post_training.identity_path_b.testing_support import ExecutionBoundary


@pytest.mark.parametrize("export_result", ["success", "preempted", "unaudited", "changed_source"])
def test_export_only_recovery_preserves_evidence_and_native_until_audit(tmp_path, monkeypatch, export_result):
    root = "s3://marin-rno2a/tmp/ttl=14d/bfcl-ablation-fixture/A1/run"
    recovery = "s3://marin-rno2a/tmp/ttl=14d/bfcl-ablation-recovery-fixture/A1/run"

    def local(path):
        destination = tmp_path / "storage" / str(path).removeprefix("s3://")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return StoragePath(str(destination))

    for module in (ablation_launch, ablation_recovery, async_launch, export_preflight, worker):
        monkeypatch.setattr(module, "StoragePath", local)
    event = CheckpointEvent(
        36,
        root + "/checkpoints/step-35",
        root + "/hf/updates-36",
        root + "/eval/updates-36",
        {},
        "a" * 40,
        "b" * 64,
        "A1",
    )
    training = dict(
        run_root=root,
        run_id="fixture",
        arm="A1",
        training_pin=event.training_pin,
        packed_sha256=event.packed_data_sha256,
        exposure_path=root + "/exposure.json",
        runtime_bytes=b'{"xla_precision":"default"}',
        runtime_sha256="c" * 64,
        source_receipt=SOURCE_RECEIPT,
        source_objects={},
    )
    exposure_raw = b'{"cumulative_batches":[{}]}'
    training["exposure_sha256"] = hashlib.sha256(exposure_raw).hexdigest()
    local(training["exposure_path"]).write_bytes(exposure_raw)
    result = {
        "arm": "A1",
        "completed_updates": 36,
        "updates": [{}] * 36,
        "training_pin": event.training_pin,
        "packed_sha256": training["packed_sha256"],
        "exposure_sha256": training["exposure_sha256"],
        "special_rows_frozen": False,
        "fully_frozen_hf_names": sorted(frozen_names(AblationArm.TRAIN_SPECIAL_ROWS)),
    }
    metadata = b'{"step":35,"is_temporary":true}'
    result_raw = json.dumps(result).encode()
    source_raw = b'{"original_source_receipt":"unchanged"}'
    local(root + "/final-checkpoint-event.json").write_bytes(event_bytes(event))
    local(root + "/training-result.json").write_bytes(result_raw)
    local(event.native_checkpoint + "/metadata.json").write_bytes(metadata)
    local(event.hf_export + "/" + ablation_recovery.SOURCE_BIND).write_bytes(source_raw)
    old_probe = export_preflight.export_write_probe(event)
    old_probe_raw = local(old_probe["path"]).read_bytes()
    evidence = dict(
        event_sha256=hashlib.sha256(event_bytes(event)).hexdigest(),
        training_result_sha256=hashlib.sha256(result_raw).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        source_bind_sha256=hashlib.sha256(source_raw).hexdigest(),
        source_bind_bytes=len(source_raw),
    )
    outcome = Future()
    with journey_world(tmp_path / "controller", monkeypatch) as world:
        parent = world.submit("export-only-cpu-owner")
        world.settle()
        execution = ExecutionBoundary(world, parent)
        monkeypatch.setattr(ablation_recovery, "native_submit", execution.submit)

        def run():
            try:
                ablation_recovery.export_only_worker(
                    training=training, event=event, recovery_root=recovery, evidence=evidence
                )
                outcome.set_result(None)
            except BaseException as error:
                outcome.set_exception(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        export = execution.get("export-36")
        assert set(execution.jobs) == {"export-36"}  # No train child in the recovery.
        assert export.kwargs["event"] == event
        assert export.submit_options["lifetime"] == 7200
        assert local(recovery + "/preserved-source-bind.json").read_bytes() == source_raw
        assert local(recovery + "/preflight/export-write-probe.bin").size() == 32
        assert local(old_probe["path"]).read_bytes() == old_probe_raw
        assert not outcome.done() and world.job(parent).state == job_pb2.JOB_STATE_RUNNING
        if export_result in ("success", "changed_source"):
            local(event.hf_export + "/path-b-checkpoint-event.json").write_bytes(event_bytes(event))
            names = frozen_names(AblationArm.TRAIN_SPECIAL_ROWS)
            audit = {
                "schema": "path_b_storage_v1",
                "source_receipt_sha256": SOURCE_RECEIPT["sha256"],
                "frozen_native_tensors": 104,
                "trained_master_cast_tensors": 398,
                "tensors": {
                    **{name: {"frozen_source": True} for name in names},
                    **{f"trained.{i}": {"frozen_source": False} for i in range(398)},
                },
            }
            local(event.hf_export + "/path-b-storage-audit.json").write_bytes(json.dumps(audit).encode())
        if export_result == "changed_source":
            local(event.hf_export + "/" + ablation_recovery.SOURCE_BIND).write_bytes(b"changed")
        execution.finish("export-36", success=export_result != "preempted")
        if export_result == "success":
            outcome.result(timeout=10)
            assert not local(event.native_checkpoint).exists()
            assert json.loads(local(recovery + "/bfcl-ready.json").read_bytes())["event"] == event.payload()
        else:
            expected_error = {"preempted": RuntimeError, "unaudited": FileNotFoundError, "changed_source": ValueError}[
                export_result
            ]
            with pytest.raises(expected_error):
                outcome.result(timeout=10)
            assert local(event.native_checkpoint + "/metadata.json").read_bytes() == metadata
            assert not local(recovery + "/bfcl-ready.json").exists()
        assert local(root + "/final-checkpoint-event.json").read_bytes() == event_bytes(event)
        assert local(root + "/training-result.json").read_bytes() == result_raw
        assert local(recovery + "/preserved-source-bind.json").read_bytes() == source_raw
        assert local(old_probe["path"]).read_bytes() == old_probe_raw
        thread.join(timeout=10)
        assert not thread.is_alive()
