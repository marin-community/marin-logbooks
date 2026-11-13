"""Score the held-out slice with the frozen filter and draw a blind labelling sheet.

Documents also present in the development sample are dropped. The sheet lists
shuffled documents with excerpts and no verdicts; the key holds the verdicts.

usage: val_prepare.py <junk_filter.py> <validation dir> <seen dir,seen dir,...> <out dir> [seed] [prefix] [junk,data,keep counts]
"""

import collections
import glob
import gzip
import importlib.util
import json
import random
import sys

spec = importlib.util.spec_from_file_location("junk_filter", sys.argv[1])
jf = importlib.util.module_from_spec(spec)
sys.modules["junk_filter"] = jf
spec.loader.exec_module(jf)
validation_dir, development_dir, out_dir = sys.argv[2:5]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
prefix = sys.argv[6] if len(sys.argv) > 6 else "validation"
STRATUM = "gt_64k"
DRAW = dict(zip(("junk", "data", "keep"), (int(v) for v in (sys.argv[7] if len(sys.argv) > 7 else "40,40,80").split(","))))
EXCERPT = 150


def read(directory):
    for path in sorted(glob.glob(f"{directory}/{STRATUM}-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def tier(category):
    if category is None:
        return "keep"
    return "junk" if category in jf.MACHINE_JUNK else "data"


seen = {(row["source"], row["doc_index"]) for directory in development_dir.split(",") for row in read(directory)}
rows = []
dropped = 0
for row in read(validation_dir):
    if (row["source"], row["doc_index"]) in seen:
        dropped += 1
        continue
    rows.append(row)
print(f"held-out documents: {len(rows)} ({dropped} dropped as already seen)")

source_tokens = collections.Counter()
for row in rows:
    source_tokens[row["source"]] += row["tokens"]
preds = []
for i, row in enumerate(rows):
    verdict = jf.junk_verdict(window["text"] for window in row["windows"])
    preds.append(
        {
            "i": i,
            "source": row["source"],
            "doc_index": row["doc_index"],
            "tokens": row["tokens"],
            "position_weight": row["share_of_replay"] * row["tokens"] / source_tokens[row["source"]],
            "category": verdict.category,
            "tier": tier(verdict.category),
            "junk_share": round(verdict.junk_share, 3),
        }
    )
with open(f"{out_dir}/{prefix}_pred.jsonl", "w") as handle:
    for p in preds:
        handle.write(json.dumps(p) + "\n")

rng = random.Random(seed)
picked = []
for name, count in DRAW.items():
    pool = [p for p in preds if p["tier"] == name]
    picked.extend(rng.sample(pool, count))
rng.shuffle(picked)
with open(f"{out_dir}/{prefix}_key.jsonl", "w") as key, open(f"{out_dir}/{prefix}_sheet.txt", "w") as sheet:
    for number, p in enumerate(picked):
        key.write(json.dumps({"number": number, **p}) + "\n")
        row = rows[p["i"]]
        sheet.write(f"\n=== {number} tokens={row['tokens']:,}\n")
        windows = row["windows"]
        chosen = windows if len(windows) <= 4 else [windows[0], windows[2], windows[4], windows[6]]
        for n, window in enumerate(chosen):
            text = window["text"]
            start = 0 if n == 0 else max(0, len(text) // 2 - EXCERPT // 2)
            sheet.write(f"  {text[start:start + EXCERPT]!r}\n")
print(f"sheet: {len(picked)} documents")
