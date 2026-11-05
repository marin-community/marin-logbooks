"""Sample the first row group of one tokenized shard per store source; count stray/duplicate thinking-close tags."""
import json, re, sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from rigging.filesystem.buckets import filesystem_for
tok = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
info = json.load(open(sys.argv[1]))
deps = [d for d in info["dependencies"] if "/datakit/tokenize/" in d]
CLOSE = re.compile(r"</(?:think|thinking|thought|reasoning)>")
OPEN = re.compile(r"<(?:think|thinking|thought|reasoning)>")
RECLOSE = re.compile(r"</(?:think|thinking|thought|reasoning)>\s*(?:(?!```)[^\s<])(?:(?!```).){20,}?</(?:think|thinking|thought|reasoning)>", re.DOTALL)
CLOSE_PROSE = re.compile(r"</(?:think|thinking|thought|reasoning)>\s*[A-Za-z]")
def scan(dep):
    name = dep.split("/datakit/tokenize/")[1].rsplit("_", 1)[0]
    fs, root = filesystem_for(dep)
    files = sorted(f for f in fs.find(root) if f.endswith(".parquet"))
    if not files:
        return name, None
    with fs.open(files[0], "rb") as h:
        pf = pq.ParquetFile(h)
        rows = []
        for rg in range(pf.num_row_groups):
            rows.extend(pf.read_row_group(rg, columns=["input_ids"]).column("input_ids").to_pylist())
            if len(rows) >= 1500:
                break
    rows = rows[:1500]
    texts = tok.batch_decode(rows)
    ntok = sum(len(r) for r in rows)
    closes = sum(len(CLOSE.findall(t)) for t in texts); opens = sum(len(OPEN.findall(t)) for t in texts)
    d_close = sum(1 for t in texts if CLOSE.search(t)); d_re = sum(1 for t in texts if RECLOSE.search(t)); d_prose = sum(1 for t in texts if CLOSE_PROSE.search(t))
    return name, {"docs": len(texts), "tokens": ntok, "closes": closes, "opens": opens, "docs_close": d_close, "docs_reclose": d_re, "docs_close_prose": d_prose, "shards": len(files)}
with ThreadPoolExecutor(8) as pool:
    results = dict(pool.map(scan, deps))
json.dump(results, open(sys.argv[2], "w"), indent=1)
print(f"{'source':44} {'docs':>5} {'close%':>7} {'reclose%':>8} {'close+prose%':>12} {'opens/closes':>12}")
for name, r in sorted(results.items(), key=lambda kv: -(kv[1] or {}).get("docs_close", 0) / max(1, (kv[1] or {}).get("docs", 1))):
    if not r or r["docs_close"] == 0:
        continue
    print(f"{name:44} {r['docs']:5d} {r['docs_close']/r['docs']:7.2%} {r['docs_reclose']/r['docs']:8.2%} {r['docs_close_prose']/r['docs']:12.2%} {r['opens']:5d}/{r['closes']:<6d}")
print("sources with no close tags in sample:", sum(1 for r in results.values() if r and r["docs_close"] == 0), "of", len(results))
