# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json

from run import run_panel


class RecordingClient:
    max_requests = 1000
    max_seconds = 900
    fingerprint = "fixture"

    def __init__(self):
        self.requests_sent = 0

    def verify_model(self):
        pass

    def score(self, prefix, target, single_token):
        self.requests_sent += 1
        return {"logprob": -len(target), "target_ids": [1]}


def test_independent_clients_preserve_the_same_complete_matrix(tmp_path):
    sequences = {
        "+": {"reference": "A" * 255, "alternate": "A" * 127 + "C" + "A" * 127},
        "-": {"reference": "T" * 255, "alternate": "T" * 127 + "G" + "T" * 127},
    }
    rows = [
        {
            "id": str(i),
            "answer_mapping": "primary" if i % 2 else "reversed",
            "chrom": "3",
            "label": i == 0,
            "sequences": sequences,
        }
        for i in range(10)
    ]
    demos = [dict(rows[i], chrom="1") for i in (0, 1)]
    panels = tmp_path / "panels.json"
    panels.write_text(
        json.dumps(
            {
                "protocol_version": 3,
                "answer_mapping_seed": 11181,
                "seed": 11181,
                "panels": {"pilot": rows, "demonstrations": demos},
            }
        )
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "panels_sha256": hashlib.sha256(panels.read_bytes()).hexdigest(),
                "matching_windows": 12,
                "unique_windows": 12,
            }
        )
    )
    outputs = []
    for workers in (1, 4):
        output = io.StringIO()
        run_panel([RecordingClient() for _ in range(workers)], panels, reference, "pilot", 10, output)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        assert records[-1]["type"] == "complete"
        assert records[-1]["requests_sent"] == 160
        scores = sorted(json.dumps(record, sort_keys=True) for record in records if record["type"] == "score")
        assert len(scores) == len(set(scores)) == 160
        outputs.append(scores)
    assert outputs[0] == outputs[1]
