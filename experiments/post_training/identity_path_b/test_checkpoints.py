# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading

from experiments.post_training.identity_path_b.checkpoints import AsyncCheckpoints, CheckpointEvent
from experiments.post_training.identity_path_b.exposure import checkpoint_cadence


def test_committed_snapshot_persists_before_independent_submission_finishes(tmp_path):
    entered, release = threading.Event(), threading.Event()
    results = []

    def submit(payload):
        entered.set()
        assert release.wait(10)
        return {"export": "independent-export", "eval": "independent-eval", "step": payload["completed_updates"]}

    hook = AsyncCheckpoints(
        write_event=lambda name, raw: (tmp_path / name).write_bytes(raw),
        submit_export_eval=submit,
        write_submission_result=lambda name, result: results.append((name, result)),
    )
    event = CheckpointEvent(
        50,
        "s3://marin-rno2a/run/checkpoints/step-49",
        "s3://marin-rno2a/run/hf/updates-50",
        {"identity_final_text_tokens": 125001},
        "reviewed-training-pin",
        "packed-sha",
        "identity10",
    )
    try:
        name = hook.publish_committed(event)
        assert entered.wait(10)
        assert json.loads((tmp_path / name).read_bytes())["exposure"]["identity_final_text_tokens"] == 125001
        assert not results  # The training-side publish returned while submission is blocked.
    finally:
        release.set()
        hook.finish_submissions()
    assert results[0][1]["submitted"]
    assert results[0][1]["jobs"]["eval"] == "independent-eval"


def test_quarter_and_identity_epoch_checkpoints_coalesce_at_completed_updates():
    assert checkpoint_cadence(100, [18, 50, 76, 100]) == [18, 25, 50, 75, 76, 100]
