# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import json
from dataclasses import replace

import pytest
from iris.rpc import job_pb2

from experiments.post_training.identity_path_b import launch
from experiments.post_training.identity_path_b.async_launch import ExportBinding, TrendInputs
from experiments.post_training.identity_path_b.workflow import ROOT_LIFETIME, workflow_worker


def inputs(monkeypatch):
    runtime = json.dumps({"xla_precision": "default"}).encode()
    sha = hashlib.sha256(runtime).hexdigest()
    # The production numerical receipt is separately bound by hash; this unit
    # fixture exercises native root serialization/policy without local notes.
    monkeypatch.setattr(launch, "RUNTIME_SHA256", sha)
    root = "s3://marin-rno2a/tmp/ttl=14d/path-b/fixture"
    binding = ExportBinding(
        "fixture",
        runtime,
        sha,
        launch.SOURCE_RECEIPT,
        launch.SOURCE_OBJECTS,
        TrendInputs(root + "/identity.jsonl", root + "/controls.jsonl"),
    )
    training = dict(
        run_root=root,
        run_id="fixture",
        arm="10",
        training_pin="a" * 40,
        packed_path=root + "/packed.parquet",
        packed_sha256="b" * 64,
        exposure_path=root + "/exposure.json",
        exposure_sha256="c" * 64,
        runtime_bytes=runtime,
        runtime_sha256=sha,
        source_receipt=launch.SOURCE_RECEIPT,
        source_objects=launch.SOURCE_OBJECTS,
    )
    return training, binding


@pytest.mark.parametrize("variant", ["bbase", "btraces"])
def test_root_submission_serializes_exact_bindings_without_credentials_and_uses_cpu_policy(monkeypatch, variant):
    training, binding = inputs(monkeypatch)
    request = launch.submission(
        name=f"pathb-{variant}-10-fixture",
        training=training,
        binding=binding,
        credentials={"CW_KEY_ID": "fake-id", "CW_KEY_SECRET": "fake-private-value"},
    )
    fn, args, kwargs = request["entrypoint"].resolve()
    inspect.signature(fn).bind(*args, **kwargs)
    assert fn is workflow_worker and not args
    assert kwargs == {"training": training, "binding": binding}
    assert b"fake-private-value" not in request["entrypoint"].workdir_files["_callable.pkl"]
    assert request["replicas"] == 1 and request["resources"].device is None
    assert request["timeout"].to_seconds() == ROOT_LIFETIME
    assert request["max_retries_failure"] == request["max_retries_preemption"] == 0
    assert request["priority_band"] == job_pb2.PRIORITY_BAND_INTERACTIVE
    assert request["environment"].env_vars["JAX_PLATFORMS"] == "cpu"
    assert list(request["environment"].extras) == ["gpu"]


def test_root_rejects_mixed_arm_non_ttl_inputs_and_source_drift(monkeypatch):
    training, binding = inputs(monkeypatch)
    creds = {"CW_KEY_ID": "fake-id", "CW_KEY_SECRET": "fake-private-value"}
    with pytest.raises(ValueError, match="Root name and arm"):
        launch.submission(name="pathb-bbase-25-fixture", training=training, binding=binding, credentials=creds)
    with pytest.raises(ValueError, match="TTL"):
        launch.submission(
            name="pathb-bbase-10-fixture",
            training={**training, "packed_path": "s3://marin-rno2a/permanent"},
            binding=binding,
            credentials=creds,
        )
    with pytest.raises(ValueError, match="source audit"):
        launch.submission(
            name="pathb-bbase-10-fixture",
            training=training,
            binding=replace(binding, source_objects={}),
            credentials=creds,
        )
