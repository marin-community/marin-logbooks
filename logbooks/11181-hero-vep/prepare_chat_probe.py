# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare six matched formatting diagnostics without making model requests."""

import argparse
import hashlib
import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer

from prompts import classification_prompt


def prepare(panels_path, tokenizer_path):
    panels = json.loads(panels_path.read_text())["panels"]
    row = panels["pilot"][0]
    assert row["chrom"] not in ("X", "chrX", 23, "23")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    path = Path(__file__).parent / "templates/marin-chat-pinned.jinja"
    template_text = path.read_text().strip()
    # HF generation tags annotate token masks; they emit no text. No assistant
    # messages are rendered here, only a user message and the generation header.
    render_text = template_text.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    template = Environment(undefined=StrictUndefined).from_string(render_text)
    result = {"variant": row["id"], "strand": "+", "mapping": row["answer_mapping"],
              "template_sha256": hashlib.sha256(template_text.encode()).hexdigest(), "cases": []}
    for method, demos in (("zero-shot", []), ("two-shot", panels["demonstrations"])):
        raw = classification_prompt(row, demos, "+", row["answer_mapping"])
        assert raw.endswith("\nAnswer:")
        user = raw.removesuffix("\nAnswer:")
        chat = template.render(messages=[{"role": "user", "content": user}],
                               bos_token="<|begin_of_text|>", add_generation_prompt=True)
        assert chat.startswith("<|begin_of_text|><|start_header_id|>user")
        assert chat.endswith("<|start_header_id|>assistant<|end_header_id|>\n")
        for arm, prompt, candidates, add_special in (
            ("raw", raw, (" A", " B"), True),
            ("chat_prefill", chat + "Answer:", (" A", " B"), False),
            ("chat_native", chat, ("A", "B"), False),
        ):
            ids = tokenizer.encode(prompt, add_special_tokens=add_special).ids
            assert ids.count(128000) == 1
            candidate_ids = []
            for answer in candidates:
                full = tokenizer.encode(prompt + answer, add_special_tokens=add_special).ids
                assert full[:len(ids)] == ids and len(full) == len(ids) + 1
                candidate_ids.append(full[-1])
            result["cases"].append({"method": method, "arm": arm, "prompt": prompt,
                "candidate_texts": candidates, "candidate_ids": candidate_ids,
                "request_without_model": {"prompt": ids, "add_special_tokens": False,
                    "max_tokens": 2, "temperature": 0, "echo": False, "logprobs": 1,
                    "logprob_token_ids": list(dict.fromkeys(candidate_ids + [220,128001,128009])),
                    "ignore_eos": True, "skip_special_tokens": False,
                    "return_tokens_as_token_ids": True}})
    assert len(result["cases"]) == 6
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("panels", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = prepare(args.panels, args.tokenizer)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: row[k] for k in ("method", "arm", "candidate_ids")} for row in result["cases"]]))
