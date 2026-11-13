# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Small replay diagnostics; authentication and endpoint configuration stay private."""

import json
import time


def mochi_probe(client, scores_path, wire_path):
    """Replay the first forward query per mapping under both prompt methods."""
    scores = [json.loads(line) for line in scores_path.open()]
    wire = [json.loads(line) for line in wire_path.open()]
    selected = {}
    results = []
    for row in scores:
        if row.get("candidate") != "A" or row.get("strand") != "+":
            continue
        mapping = row["mapping"]
        selected.setdefault(mapping, row["variant"])
        if row["variant"] != selected[mapping]:
            continue
        old = wire[row["paired_request"] - 1]
        assert old["route"] == "v1/completions"
        request = dict(old["request"])
        assert request["logprob_token_ids"] == [362, 426]
        request.pop("logprob_token_ids")
        request.update(logprobs=20, return_tokens_as_token_ids=False)
        response = client.request("v1/completions", request)
        assert response["system_fingerprint"] == client.fingerprint
        results.append({"variant": row["variant"], "method": row["method"], "mapping": mapping,
                        "previous_A_logprob": row["logprob"], "response": response})
        time.sleep(1)
    assert len(results) == 4
    return results


def openrouter_probe(session):
    """Three requests: output-logprob control and two input-logprob interfaces."""
    base = {"model": "z-ai/glm-5.3", "provider": {"only": ["fireworks"],
            "allow_fallbacks": False, "require_parameters": True},
            "max_tokens": 1, "temperature": 1}
    dna = "ACGTACGTGACTGACTACGT"
    cases = [
        ("output_control", "chat/completions", {"messages": [{"role": "user", "content": "Reply with A."}],
          "logprobs": True, "top_logprobs": 5}),
        ("completion_echo", "completions", {"prompt": dna, "echo": True, "logprobs": 5}),
        ("prompt_logprobs", "completions", {"prompt": dna, "echo": True, "logprobs": 5,
          "prompt_logprobs": 5}),
    ]
    results = []
    for name, route, extra in cases:
        payload = {**base, **extra}
        response = session.post("https://openrouter.ai/api/v1/" + route, json=payload,
                                timeout=45, allow_redirects=False)
        results.append({"case": name, "request": payload, "status": response.status_code,
                        "response": response.json()})
    return results
