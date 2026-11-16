# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Four-variant format/mapping replication with eight end-of-run repeats."""

import json
import time
from pathlib import Path

from tokenizers import Tokenizer

from prepare_chat_probe import prepare


def prepare_replication(panels_path: Path, tokenizer_path: Path) -> dict:
    rows = json.loads(panels_path.read_text())["panels"]["pilot"]
    groups = list(dict.fromkeys(row["match_group"] for row in rows))[:2]
    indices = []
    for group in groups:
        for label in (False, True):
            indices.append(next(i for i, row in enumerate(rows)
                                if row["match_group"] == group and row["label"] == label))
    assert len(set(indices)) == 4
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cases = []
    for index in indices:
        for mapping in ("primary", "reversed"):
            prepared = prepare(panels_path, tokenizer_path, query_index=index, mapping=mapping, tokenizer=tokenizer)
            for case in prepared["cases"]:
                if case["arm"] == "chat_native":
                    continue
                cases.append({"variant": prepared["variant"], "mapping": mapping,
                              "label": rows[index]["label"], "match_group": rows[index]["match_group"],
                              "repeat": False, **case})
    assert len(cases) == 32
    cases.extend([{**case, "repeat": True} for case in cases[:8]])
    return {"selection": "First two frozen pilot groups; first control and positive in each; forward strand",
            "cases": cases, "max_completions": 40, "max_http_requests": 137, "max_seconds": 600}


def run_replication(client, plan, record):
    for index, case in enumerate(plan["cases"]):
        req = case["request_without_model"]
        ids = req["prompt"]
        if not case["repeat"]:
            for suffix, expected in [("", ids)] + [
                (answer, ids + [token]) for answer, token in zip(case["candidate_texts"], case["candidate_ids"])
            ]:
                tokenized = client.request("tokenize", {"model": client.model, "prompt": case["prompt"] + suffix,
                                            "add_special_tokens": case["arm"] == "raw"})
                assert tokenized["tokens"] == expected, "Live tokenizer mismatch"
        response = client.request("v1/completions", {"model": client.model, **req})
        assert response["model"] == client.model
        if client.fingerprint is None:
            client.fingerprint = response["system_fingerprint"]
        assert response["system_fingerprint"] == client.fingerprint, "Serving build changed"
        assert response["usage"]["prompt_tokens"] == len(ids)
        assert response["usage"]["completion_tokens"] == 2
        record({"index": index, **{k: case[k] for k in
                ("variant", "mapping", "label", "match_group", "method", "arm", "repeat", "candidate_ids")},
                "response": response})
        time.sleep(1)
