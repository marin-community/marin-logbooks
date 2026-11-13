"""Extract the held-out false positives with per-window chunk labels and the features behind them.

usage: fp_extract.py <junk_filter.py> <out dir> <validation dir> <out.json>
"""

import dataclasses
import glob
import gzip
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("junk_filter", sys.argv[1])
jf = importlib.util.module_from_spec(spec)
sys.modules["junk_filter"] = jf
spec.loader.exec_module(jf)
out_dir, validation_dir = sys.argv[2], sys.argv[3]

labels = {}
for line in open(f"{out_dir}/validation_labels.txt"):
    if line.strip() and not line.startswith("#"):
        number, label = line.split()
        labels[int(number)] = label
wanted = {}
for line in open(f"{out_dir}/validation_key.jsonl"):
    k = json.loads(line)
    if k["tier"] != "keep" and labels[k["number"]] == "K":
        wanted[k["source"], k["doc_index"]] = k

KEEP = ("compression_ratio", "separator_share", "templated_line_share", "repeated_line_share", "delimited_line_share",
        "key_value_line_share", "digit_share", "letter_share", "mean_line_chars", "lines", "markup_share", "syntax_share",
        "code_line_share", "item_line_share")
docs = []
for path in sorted(glob.glob(f"{validation_dir}/gt_64k-*.jsonl.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            k = wanted.get((row["source"], row["doc_index"]))
            if k is None:
                continue
            windows = []
            counts = {}
            for window in row["windows"]:
                chunks = []
                for chunk in jf.chunks([window["text"]]):
                    f = jf.chunk_features(chunk)
                    category = jf.features_junk_category(f)
                    counts[str(category)] = counts.get(str(category), 0) + 1
                    values = dataclasses.asdict(f)
                    chunks.append({"label": category, "features": {name: round(values[name], 3) for name in KEEP}, "text": chunk[:900]})
                windows.append({"start": window["start"], "chunks": chunks})
            docs.append(
                {
                    "number": k["number"],
                    "source": row["source"],
                    "doc_index": row["doc_index"],
                    "tokens": row["tokens"],
                    "category": k["category"],
                    "junk_share": k["junk_share"],
                    "counts": counts,
                    "windows": windows,
                }
            )
docs.sort(key=lambda d: d["number"])
json.dump(docs, open(sys.argv[4], "w"), ensure_ascii=False)
for d in docs:
    print(d["number"], d["source"], d["doc_index"], f"{d['tokens']:,}", d["category"], d["junk_share"], d["counts"])
    for w in d["windows"]:
        for c in w["chunks"]:
            if c["label"]:
                f = c["features"]
                print("   ", w["start"], c["label"], {k: v for k, v in f.items() if v}, repr(c["text"][:110]))
                break
