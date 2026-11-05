"""Per-response scan: stray closes and close->prose->close (no re-open) within one assistant response."""
import json, re, sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from rigging.filesystem.buckets import filesystem_for
tok = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
NAMES = sys.argv[3].split(",")
info = json.load(open(sys.argv[1]))
deps = [d for d in info["dependencies"] if "/datakit/tokenize/" in d and d.split("/datakit/tokenize/")[1].rsplit("_", 1)[0] in NAMES]
TAG = re.compile(r"<(/?)(think|thinking|thought|reasoning)>")
BOUNDARY = re.compile(r"</?(?:assistant|user|system|tool|tool_call[^>]*|tool_response|observation)>|<\|[a-z_]+\|>|\n(?:THOUGHT:|Observation:)", re.IGNORECASE)
def analyze(text):
    segs = BOUNDARY.split(text)
    stray = reclose = closes = opens = segs_reclose = 0
    examples = []
    for s in segs:
        depth = 0; last_close_end = None
        for m in TAG.finditer(s):
            if m.group(1) == "":
                opens += 1; depth += 1; last_close_end = None
            else:
                closes += 1
                if depth == 0:
                    stray += 1
                    if last_close_end is not None and len(s[last_close_end:m.start()].strip()) >= 20:
                        reclose += 1
                        if len(examples) < 1: examples.append(s[max(0, last_close_end - 80):m.end() + 40])
                else:
                    depth -= 1
                last_close_end = m.end()
        if reclose: segs_reclose += 1
    return stray, reclose, closes, opens, examples
def scan(dep):
    name = dep.split("/datakit/tokenize/")[1].rsplit("_", 1)[0]
    fs, root = filesystem_for(dep)
    files = sorted(f for f in fs.find(root) if f.endswith(".parquet"))
    rows = []
    with fs.open(files[0], "rb") as h:
        pf = pq.ParquetFile(h)
        for rg in range(pf.num_row_groups):
            rows.extend(pf.read_row_group(rg, columns=["input_ids"]).column("input_ids").to_pylist())
            if len(rows) >= 1500: break
    rows = rows[:1500]
    texts = tok.batch_decode(rows)
    res = [analyze(t) for t in texts]
    ex = next((r[4][0] for r in res if r[4]), None)
    return name, {"docs": len(texts), "docs_stray": sum(1 for r in res if r[0]), "docs_reclose": sum(1 for r in res if r[1]),
                  "stray": sum(r[0] for r in res), "reclose": sum(r[1] for r in res), "closes": sum(r[2] for r in res), "opens": sum(r[3] for r in res), "example": ex}
with ThreadPoolExecutor(9) as pool:
    results = dict(pool.map(scan, deps))
json.dump(results, open(sys.argv[2], "w"), indent=1)
print(f"{'source':44} {'docs':>5} {'docs w/ stray close':>20} {'docs w/ close->prose->close':>28} {'opens/closes':>14}")
for name, r in sorted(results.items(), key=lambda kv: -kv[1]["docs_reclose"]):
    print(f"{name:44} {r['docs']:5d} {r['docs_stray']/r['docs']:20.2%} {r['docs_reclose']/r['docs']:28.2%} {r['opens']:6d}/{r['closes']:<7d}")
for name, r in results.items():
    if r["example"]: print(f"--- {name} example:\n{r['example']!r}")
