# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Reproduce saved serving tokenizations without loading model weights."""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def audit(tokenizer_path: Path, wire_path: Path) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result = {"tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
              "examples": [], "wire_checks": []}
    for text in ("Answer:", "Answer: ", "Answer: A", "Answer: B", "Answer:\nA", " A", "A", " "):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        result["examples"].append({"text": text, "ids": ids,
                                   "pieces": [tokenizer.decode([i], skip_special_tokens=False) for i in ids]})
    for ids in ([16533, 25, 362], [16533, 25, 220, 32]):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        result.setdefault("roundtrips", []).append({"ids": ids, "text": text,
                "reencoded_ids": tokenizer.encode(text, add_special_tokens=False).ids})
    for line in wire_path.open():
        row = json.loads(line)
        if row.get("route") != "tokenize" or row.get("status") != 200:
            continue
        request = row["request"]
        expected = json.loads(row["response_text"])["tokens"]
        ids = tokenizer.encode(request["prompt"], add_special_tokens=request["add_special_tokens"]).ids
        result["wire_checks"].append({"prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
                                    "tokens": len(expected), "exact_match": ids == expected,
                                    "matches_with_explicit_bos": [128000] + ids == expected})
        if len(result["wire_checks"]) >= 32:
            break
    result["special_tokens"] = {str(i): tokenizer.id_to_token(i) for i in (128000,128001,128005,128009,128011)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("wire", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.tokenizer, args.wire)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
