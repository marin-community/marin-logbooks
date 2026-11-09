# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-step trained-token share of each data source. Usage: source_mix.py OUT_JSONL FILES..."""
import io
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zstandard
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
SYS_HDR = tok.encode("<|start_header_id|>system<|end_header_id|>", add_special_tokens=False)
dec = [tok.decode([i]) for i in range(128000)]
ENDS_LT = np.array([i for i, x in enumerate(dec) if x.endswith("<")])
THINK = 27963


def classify(seg):
    if len(seg) < 2 or seg[0] != 128000:
        return "continuation"
    if seg[1] == 128006:
        head = tok.decode(seg[1:40].tolist())
        mode = "nothink" if "/nothink" in head else ("think" if "/think" in head else "other_sys")
        tools = "### Tools" in head
        return f"chat_{mode}{'_tools' if tools else ''}"
    head = tok.decode(seg[1:12].tolist())
    if head.startswith("<system>") or head.startswith("<user>"):
        return "plaintag_chat"
    if head.startswith("!!!"):
        return "pad"
    idx = np.nonzero(seg[1:] == THINK)[0] + 1
    if len(idx) and np.isin(seg[idx - 1], ENDS_LT).any():
        return "raw_with_think_tag"
    return "raw_text"


def scan(path):
    a = np.load(io.BytesIO(zstandard.ZstdDecompressor().decompressobj().decompress(Path(path).read_bytes())))
    t, s, w = a["tokens"].astype(np.int64), a["segment_ids"], a["loss_weight"]
    c = Counter()
    for r in range(t.shape[0]):
        starts = np.r_[0, np.nonzero(np.diff(s[r]))[0] + 1]
        ends = np.r_[starts[1:], t.shape[1]]
        for st, en in zip(starts, ends, strict=True):
            k = classify(t[r][st:en])
            c[k] += int(w[r][st:en].sum())
    return {"step": int(path.rsplit("-", 1)[1].split(".")[0]), **c}


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex, open(sys.argv[1], "w") as out:
        for res in ex.map(scan, sys.argv[2:]):
            out.write(json.dumps(res) + "\n")
            out.flush()
