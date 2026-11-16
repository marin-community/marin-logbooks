# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen 100-variant chat-prefill condition; no DNA scoring or validation panel."""

import hashlib
import json
import math
import time
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer

from prompts import classification_prompt


def prepare(panels_path: Path, tokenizer_path: Path) -> dict:
    panel_bytes = panels_path.read_bytes()
    panels = json.loads(panel_bytes)
    rows, demos = panels["panels"]["pilot"], panels["panels"]["demonstrations"]
    assert panels["protocol_version"] == 3 and len(rows) == 100
    assert len({r["id"] for r in rows}) == 100
    assert all(r["chrom"] in {str(i) for i in range(2, 23)} for r in rows)
    assert len(demos) == 2 and all(r["chrom"] == "1" for r in demos)
    tok = Tokenizer.from_file(str(tokenizer_path))
    template_text = (Path(__file__).parent / "templates/marin-chat-pinned.jinja").read_text().strip()
    template = Environment(undefined=StrictUndefined).from_string(
        template_text.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    )
    cases = []
    for row in rows:
        for strand in ("+", "-"):
            for method, examples in (("zero-shot", []), ("two-shot", demos)):
                raw = classification_prompt(row, examples, strand, row["answer_mapping"])
                assert raw.endswith("\nAnswer:")
                prompt = template.render(messages=[{"role": "user", "content": raw.removesuffix("\nAnswer:")}],
                    bos_token="<|begin_of_text|>", add_generation_prompt=True) + "Answer:"
                ids = tok.encode(prompt, add_special_tokens=False).ids
                assert ids.count(128000) == 1
                candidates = []
                for answer in (" A", " B"):
                    full = tok.encode(prompt + answer, add_special_tokens=False).ids
                    assert full[:len(ids)] == ids and len(full) == len(ids) + 1
                    candidates.append(full[-1])
                assert candidates == [362, 426]
                cases.append({"index": len(cases), "variant": row["id"], "method": method,
                    "strand": strand, "mapping": row["answer_mapping"], "prompt": prompt,
                    "prompt_ids": ids, "candidate_ids": candidates})
    return {"protocol": "chat-prefill-v1", "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
            "template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
            "answer_mapping_seed": panels["answer_mapping_seed"], "cases": cases,
            "max_http_requests": 1609, "max_seconds": 1800, "max_completion_seconds": 10,
            "repeat_after": [200, 400], "repeat_indices": [0, 1, 2, 3]}


def run(client, plan, emit):
    baselines = {}
    def score(case, repeat):
        req = {"model": client.model, "prompt": case["prompt_ids"], "add_special_tokens": False,
               "max_tokens": 1, "echo": False, "logprobs": 1,
               "logprob_token_ids": case["candidate_ids"], "temperature": 1.0, "seed": 11181,
               "ignore_eos": True, "skip_special_tokens": False, "return_tokens_as_token_ids": True}
        start = time.monotonic()
        response = client.request("v1/completions", req)
        elapsed = time.monotonic() - start
        assert response["model"] == client.model
        if client.fingerprint is None:
            client.fingerprint = response["system_fingerprint"]
        assert response["system_fingerprint"] == client.fingerprint, "Serving build changed"
        assert response["usage"]["prompt_tokens"] == len(case["prompt_ids"])
        assert response["usage"]["completion_tokens"] == 1
        choice = response["choices"][0]
        lp = choice["logprobs"]["top_logprobs"][0]
        values = [lp[f"token_id:{i}"] for i in case["candidate_ids"]]
        assert all(math.isfinite(x) for x in values)
        row = {k: case[k] for k in ("index", "variant", "method", "strand", "mapping")}
        row.update(type="repeat" if repeat else "score", logp_A=values[0], logp_B=values[1],
            prompt_sha256=hashlib.sha256(case["prompt"].encode()).hexdigest(),
            sampled_token=choice["text"], elapsed=elapsed)
        if repeat:
            row["max_logprob_delta"] = max(abs(a-b) for a,b in zip(values, baselines[case["index"]]))
        emit(row)
        if repeat:
            assert row["max_logprob_delta"] <= 1e-5, "Repeated score drift; stop before scaling further"
        else:
            baselines[case["index"]] = values
        assert elapsed <= plan["max_completion_seconds"], "Slow completion; shared endpoint load guard"
        time.sleep(1)
    emit({"type": "start", "protocol": plan["protocol"], "panel_sha256": plan["panel_sha256"]})
    for i, case in enumerate(plan["cases"], 1):
        for suffix, expected in [("", case["prompt_ids"])] + [
            (a, case["prompt_ids"]+[t]) for a,t in zip((" A", " B"),case["candidate_ids"])
        ]:
            result = client.request("tokenize", {"model": client.model, "prompt": case["prompt"]+suffix,
                                                "add_special_tokens": False})
            assert result["tokens"] == expected, "Live tokenization changed"
        score(case, False)
        if i in plan["repeat_after"]:
            for index in plan["repeat_indices"]:
                score(plan["cases"][index], True)
    emit({"type": "complete", "classification_responses": 400, "repeat_responses": 8})
