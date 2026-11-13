# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
from dataclasses import asdict

import pytest
from classification_rerun import load_reused_dna, run_classification_rerun
from targets import targets


def test_reuse_rejects_changed_inputs_and_runner_never_rescores_dna(tmp_path):
    rows = [
        {
            "id": f"2:{i}:A:C",
            "chrom": "2",
            "label": i % 10 == 0,
            "match_group": i // 10,
            "answer_mapping": "primary" if i % 2 else "reversed",
            "sequences": {s: {"reference": "A" * 255, "alternate": "A" * 127 + "C" + "A" * 127} for s in ("+", "-")},
        }
        for i in range(100)
    ]
    demos = [{**r, "chrom": "1"} for r in rows[:2]]
    panels = {
        "protocol_version": 3,
        "answer_mapping_seed": 11181,
        "panels": {"pilot": rows, "validation": [], "demonstrations": demos},
    }
    path = tmp_path / "panels.json"
    old_path = tmp_path / "old.json"
    reference_path = tmp_path / "reference.json"
    scores_path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(panels))
    old_path.write_text(json.dumps(panels))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reference_path.write_text(json.dumps({"panels_sha256": digest, "matching_windows": 402, "unique_windows": 402}))
    records = [{"type": "start", "panels_sha256": digest, "panel": "pilot", "variant_count": 100}]
    for t in targets(rows, demos):
        if t.single_token:
            continue
        r = asdict(t)
        r["input_sha256"] = hashlib.sha256((t.prefix + t.text).encode()).hexdigest()
        del r["prefix"], r["text"]
        records.append({"type": "score", **r, "logprob": -1.0})
    records.append({"type": "complete"})
    scores_path.write_text("\n".join(json.dumps(r) for r in records))

    class Client:
        fingerprint = "fixture"
        max_requests = 2000
        max_seconds = 1800
        requests_sent = 0
        session = type("Session", (), {"headers": {}})()

        def verify_model(self):
            pass

        def check_sentinels(self):
            assert self.baseline == [-1.0, -1.0]

        def score(self, prefix, target, single_token):
            assert single_token and target in {" A", " B"}
            self.requests_sent += 1
            return {"logprob": -0.5}

    out = io.StringIO()
    client = Client()
    run_classification_rerun(client, path, reference_path, old_path, scores_path, out)
    result = [json.loads(line) for line in out.getvalue().splitlines()]
    assert result[-1]["type"] == "complete" and client.requests_sent == 800
    assert sum(r.get("origin") == "historical_dna" for r in result) == 800
    rows[0]["sequences"]["+"]["reference"] = "G" * 255
    path.write_text(json.dumps(panels))
    with pytest.raises(AssertionError, match="changed"):
        load_reused_dna(path, old_path, scores_path)
