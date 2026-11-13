"""Show a seeded random sample of window-sample documents with one predicted category.

usage: show2.py <sample dir> <stratum> <pred.jsonl> <category|None|any> [count] [excerpt_chars] [seed] [windows_to_show]
"""

import glob
import gzip
import json
import random
import sys

stratum, wanted = sys.argv[2], sys.argv[4]
count = int(sys.argv[5]) if len(sys.argv) > 5 else 12
width = int(sys.argv[6]) if len(sys.argv) > 6 else 110
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
shown = int(sys.argv[8]) if len(sys.argv) > 8 else 3
preds = [json.loads(line) for line in open(sys.argv[3])]
pool = [p for p in preds if wanted == "any" or str(p["category"]) == wanted]
random.Random(seed).shuffle(pool)
pick = {p["i"]: p for p in pool[:count]}
print(f"{len(pool)} documents predicted {wanted}; showing {len(pick)}")
i = 0
for path in sorted(glob.glob(f"{sys.argv[1]}/{stratum}-*.jsonl.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if i in pick:
                row = json.loads(line)
                p = pick[i]
                print(f"\n#{i} {row['source']} di={row['doc_index']} tokens={row['tokens']:,} -> {p['category']} share={p['junk_share']} {p['counts']}")
                windows = row["windows"]
                step = max(1, len(windows) // shown)
                for window in windows[::step][:shown]:
                    text = window["text"]
                    middle = len(text) // 2
                    print(f"  @{window['start']:>11,}: {text[middle:middle + width]!r}")
            i += 1
