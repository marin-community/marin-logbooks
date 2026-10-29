# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest

from experiments.post_training.identity_fullscreen_owner import run_unseen
from experiments.post_training.identity_fullscreen_owner.math_signature import pinned_signature_smoke
from experiments.post_training.identity_fullscreen_owner.run_unseen import UNSEEN_DIR, checked_unseen_rows


def test_committed_unseen_pair_loads_exact_earlier_thirty_including_empty_identity():
    rows = checked_unseen_rows()
    assert (UNSEEN_DIR / "identity.jsonl").read_bytes() == b""
    assert [row.id for row in rows] == [f"tp-unseen-{i:03d}" for i in range(30)]


@pytest.mark.manual
def test_wrapper_calls_real_f5_keyword_only_signature_and_stops_before_rpc():
    archive = Path(os.environ["PATHB_F5_ARCHIVE"])
    report = pinned_signature_smoke(str(archive), "2f0d5ce76bb408f0e803bd51044472af11ddb77b364443d9b09c1a5f5ab7379d")
    assert report["real_f5_signature"] == "PASS"
    assert report["model_calls"] == report["child_submissions"] == 0


def test_unseen_continuation_issues_one_unseen_batch_and_no_dev_regeneration(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run_unseen, "require_secret", lambda name: "http://unused.invalid/v1")
    monkeypatch.setattr(run_unseen.shared.generate, "callback", lambda **kwargs: calls.append(kwargs))
    output = tmp_path / "continuation"
    monkeypatch.setattr(sys, "argv", ["run_unseen", "--label", "fixture", "--model", "fixture", "--out", str(output)])
    run_unseen.main()
    assert len(calls) == 1
    assert calls[0]["prompt_dir"] == UNSEEN_DIR
    assert calls[0]["out"] == output / "unseen-third-person"
    assert calls[0]["conditions"] == ("no_system", "system_baseline")
    assert calls[0]["workers"] == 16 and calls[0]["split"] == "dev"
    assert not (output / "generate").exists()
