# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from concurrent.futures import Future

import pytest
from iris.rpc import job_pb2
from iris.testing.journeys.world import journey_world
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.identity_path_b import ablation_launch, async_launch
from experiments.post_training.identity_path_b.ablation import AblationArm, frozen_names, updates_for
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent, event_bytes
from experiments.post_training.identity_path_b.launch import SOURCE_RECEIPT
from experiments.post_training.identity_path_b.testing_support import ExecutionBoundary


@pytest.mark.parametrize("arm", [AblationArm.CONTROL, AblationArm.GLM_ON_TENTH_72, AblationArm.IDENTITY_OFF_72])
@pytest.mark.parametrize("probe_failure", [False, True])
def test_final_only_owner_releases_train_before_probe_export_and_outlives_export(
    tmp_path, monkeypatch, probe_failure, arm
):
    updates = updates_for(arm)
    export_name = f"export-{updates}"
    root_uri = f"s3://marin-rno2a/tmp/ttl=14d/bfcl-ablation-fixture/{arm}"

    def local(path):
        destination = tmp_path / "storage" / str(path).removeprefix("s3://")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return StoragePath(str(destination))

    monkeypatch.setattr(ablation_launch, "StoragePath", local)
    monkeypatch.setattr(async_launch, "StoragePath", local)
    monkeypatch.setattr(
        ablation_launch, "record", lambda path, value: local(path).write_bytes(json.dumps(value).encode())
    )
    exposure = {"cumulative_batches": [{"identity_final_text_tokens": 1}] * updates}
    monkeypatch.setattr(ablation_launch, "read_pinned", lambda *args: json.dumps(exposure).encode())
    training = dict(
        run_root=root_uri,
        run_id="fixture",
        arm=arm,
        training_pin="a" * 40,
        packed_path="packed",
        packed_sha256="b" * 64,
        exposure_path="exposure",
        exposure_sha256="c" * 64,
        runtime_bytes=b'{"xla_precision":"default"}',
        runtime_sha256="d" * 64,
        source_receipt=SOURCE_RECEIPT,
        source_objects={},
    )
    event = CheckpointEvent(
        updates,
        root_uri + f"/checkpoints/step-{updates - 1}",
        root_uri + f"/hf/updates-{updates}",
        root_uri + f"/eval/updates-{updates}",
        exposure["cumulative_batches"][-1],
        "a" * 40,
        "b" * 64,
        arm,
    )
    local(root_uri + "/final-checkpoint-event.json").write_bytes(event_bytes(event))
    local(event.native_checkpoint + "/metadata.json").write_bytes(
        json.dumps({"step": updates - 1, "is_temporary": True}).encode()
    )
    local(event.hf_export + "/path-b-checkpoint-event.json").write_bytes(event_bytes(event))
    audit_names = frozen_names(AblationArm.CONTROL)
    audit = {
        "schema": "path_b_storage_v1",
        "source_receipt_sha256": SOURCE_RECEIPT["sha256"],
        "frozen_native_tensors": 104,
        "trained_master_cast_tensors": 398,
        "tensors": {
            **{name: {"frozen_source": True} for name in audit_names},
            **{f"trained.{index}": {"frozen_source": False} for index in range(398)},
        },
    }
    local(event.hf_export + "/path-b-storage-audit.json").write_bytes(json.dumps(audit).encode())
    probes = []
    outcome = Future()
    with journey_world(tmp_path / "controller", monkeypatch) as world:
        parent = world.submit("ablation-cpu-owner")
        world.settle()
        execution = ExecutionBoundary(world, parent)
        monkeypatch.setattr(ablation_launch, "native_submit", execution.submit)

        def probe(event):
            assert world.job(execution.jobs["train"].ref).state == job_pb2.JOB_STATE_SUCCEEDED
            assert export_name not in execution.jobs
            probes.append(event)
            if probe_failure:
                raise PermissionError("CPU preflight denied")
            return {"bytes": 32}

        monkeypatch.setattr(ablation_launch, "export_write_probe", probe)

        def run():
            try:
                ablation_launch.workflow_worker(training=training)
                outcome.set_result(None)
            except BaseException as error:
                outcome.set_exception(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        train = execution.get("train")
        assert train.submit_options["env_vars"]["JAX_PLATFORMS"] == "cuda,cpu"
        assert not outcome.done() and not probes
        execution.finish("train")
        if probe_failure:
            with pytest.raises(PermissionError, match="preflight denied"):
                outcome.result(timeout=10)
            assert set(execution.jobs) == {"train"}
            assert local(event.native_checkpoint).exists()
        else:
            export = execution.get(export_name)
            assert probes == [event] and not outcome.done()
            assert world.job(parent).state == job_pb2.JOB_STATE_RUNNING
            assert world.job(export.ref).state == job_pb2.JOB_STATE_RUNNING
            assert export.submit_options["lifetime"] == 7200
            execution.finish(export_name)
            outcome.result(timeout=10)
            assert not local(event.native_checkpoint).exists()
            assert (
                json.loads(local(root_uri + "/bfcl-ready.json").read_bytes())["status"]
                == "audited_HF_ready_BFCL_pending"
            )
        worker.join(timeout=10)
        assert not worker.is_alive()
