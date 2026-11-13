"""Show documents whose verdict changed between two prediction files.

usage: changed.py <sample dir> <stratum> <old pred> <new pred> [limit] [excerpt_chars]
"""

import glob
import gzip
import json
import sys

limit = int(sys.argv[5]) if len(sys.argv) > 5 else 30
width = int(sys.argv[6]) if len(sys.argv) > 6 else 100
old = [json.loads(line) for line in open(sys.argv[3])]
new = [json.loads(line) for line in open(sys.argv[4])]
changed = {n["i"]: (o["category"], n["category"], n["junk_share"]) for o, n in zip(old, new, strict=True) if o["category"] != n["category"]}
print(len(changed), "changed")
shown = 0
i = 0
for path in sorted(glob.glob(f"{sys.argv[1]}/{sys.argv[2]}-*.jsonl.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if i in changed and shown < limit:
                row = json.loads(line)
                before, after, share = changed[i]
                print(f"\n#{i} {row['source']} tokens={row['tokens']:,} {before} -> {after} share={share}")
                for window in row["windows"][1::3][:2]:
                    text = window["text"]
                    print(f"  {text[len(text) // 2:len(text) // 2 + width]!r}")
                shown += 1
            i += 1
