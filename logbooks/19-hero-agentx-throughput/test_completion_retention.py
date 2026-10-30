# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from completion_retention import retain_finished_attempt

from experiments.evaluation.agentx.artifacts import file_digest, read_record, unpack


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_finished_attempt_survives_loss_of_original_directory(tmp_path: Path, status: str) -> None:
    output = tmp_path / "output"
    point = output / "point"
    point.mkdir(parents=True)
    attempt = {
        "status": status,
        "finished_unix": 1,
        "iris_attempt_uid": "fixture-attempt",
        "iris_task_id": "/fixture/cpu-retention-test/0:0",
    }
    (point / "attempt.json").write_text(json.dumps(attempt))
    payload = bytes(range(256)) * 37
    evidence = point / "evidence.bin"
    evidence.write_bytes(payload)
    digest = file_digest(evidence)
    prefix = str(tmp_path / "retained")
    receipt = retain_finished_attempt(output, prefix, 0, "fixture-attempt")
    evidence.unlink()
    (point / "attempt.json").unlink()
    assert read_record(f"{prefix}/host-0/fixture-attempt/retention.json") == receipt
    assert receipt["application_status"] == status
    assert receipt["archive_sha256"] == receipt["readback_sha256"]
    restored = tmp_path / "restored"
    unpack(Path(receipt["archive_uri"]), restored)
    assert file_digest(restored / "point/evidence.bin") == digest
    assert (restored / "point/evidence.bin").read_bytes() == payload
    assert json.loads((restored / "point/attempt.json").read_text()) == attempt
