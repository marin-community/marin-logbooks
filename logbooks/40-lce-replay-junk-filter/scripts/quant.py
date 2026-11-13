"""Chunk-feature quantiles by review group, to place thresholds.

usage: quant.py <junk_filter.py> <replay.jsonl> <pred.jsonl> [every_nth_doc]
"""

import dataclasses
import importlib.util
import json
import re
import sys

spec = importlib.util.spec_from_file_location("junk_filter", sys.argv[1])
jf = importlib.util.module_from_spec(spec)
sys.modules["junk_filter"] = jf
spec.loader.exec_module(jf)
preds = [json.loads(line) for line in open(sys.argv[3])]
step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
MARKER = re.compile(r"\n\n\[… [\d,]+ tokens omitted …\]\n\n")
FIELDS = ["compression_ratio", "separator_share", "templated_line_share", "repeated_line_share", "syntax_share",
          "key_value_line_share", "item_line_share", "encoded_share", "digit_share", "mean_line_chars", "markup_share"]
values = {}
for i, line in enumerate(open(sys.argv[2])):
    if i % step:
        continue
    row = json.loads(line)
    group = preds[i]["group"]
    if group == "keep" and preds[i]["prior"] not in ("prose", "source_code", "prose_non_latin", "doc_mixed"):
        group = "keep_other"
    elif group == "keep":
        group = "keep_" + ("code" if preds[i]["prior"] == "source_code" else "prose")
    for chunk in jf.chunks(MARKER.split(row["text"])):
        f = dataclasses.asdict(jf.chunk_features(chunk))
        for field in FIELDS:
            values.setdefault((group, field), []).append(f[field])
QS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
for field in FIELDS:
    print(f"\n{field}   quantiles {QS}")
    for group in ("keep_prose", "keep_code", "keep_other", "data", "junk"):
        v = sorted(values[group, field])
        print(f"  {group:11s} n={len(v):6d} " + " ".join(f"{v[min(len(v) - 1, int(q * len(v)))]:8.3f}" for q in QS))
