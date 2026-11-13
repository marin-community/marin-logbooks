"""Run the junk filter over the window sample and report shares.

usage: score2.py <junk_filter.py> <sample dir> <stratum> <out_predictions.jsonl>
"""

import collections
import glob
import gzip
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("junk_filter", sys.argv[1])
jf = importlib.util.module_from_spec(spec)
sys.modules["junk_filter"] = jf
spec.loader.exec_module(jf)
stratum = sys.argv[3]
CAP = 262_144


def tier(category):
    if category is None:
        return "keep"
    return "junk" if category in jf.MACHINE_JUNK else "data"


rows = []
for path in sorted(glob.glob(f"{sys.argv[2]}/{stratum}-*.jsonl.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))

# A cache's documents stand for its share of replay; weight each document so the sample totals match.
per_source = collections.Counter(row["source"] for row in rows)
source_tokens = collections.Counter()
for row in rows:
    source_tokens[row["source"]] += row["tokens"]
out = open(sys.argv[4], "w")
docs = collections.Counter()
toks = collections.Counter()
capped = collections.Counter()
positions = collections.Counter()
total_tokens = total_capped = total_share = 0
for i, row in enumerate(rows):
    verdict = jf.junk_verdict(window["text"] for window in row["windows"])
    key = str(verdict.category)
    docs[key] += 1
    toks[key] += row["tokens"]
    capped[key] += min(row["tokens"], CAP)
    # Training positions: the cache's share of replay, split across its documents by token count.
    weight = row["share_of_replay"] * row["tokens"] / source_tokens[row["source"]]
    positions[key] += weight
    total_tokens += row["tokens"]
    total_capped += min(row["tokens"], CAP)
    total_share += weight
    out.write(
        json.dumps(
            {
                "i": i,
                "source": row["source"],
                "doc_index": row["doc_index"],
                "tokens": row["tokens"],
                "category": verdict.category,
                "tier": tier(verdict.category),
                "junk_share": round(verdict.junk_share, 3),
                "counts": {str(k): v for k, v in verdict.counts.items()},
                "chunks": verdict.chunks,
            }
        )
        + "\n"
    )
n = len(rows)
print(f"{stratum}: {n} documents from {len(per_source)} caches, {total_tokens/1e9:.2f}B tokens")
print(f"{'category':18s} {'docs':>6s} {'docs%':>7s} {'tokens%':>8s} {'capped%':>8s} {'positions%':>10s}")
for key, count in docs.most_common():
    print(
        f"{key:18s} {count:6d} {count/n:7.1%} {toks[key]/total_tokens:8.1%} {capped[key]/total_capped:8.1%} "
        f"{positions[key]/total_share:10.1%}"
    )
for name, members in (("machine junk", jf.MACHINE_JUNK), ("data dumps", jf.DATA_DUMPS)):
    keys = [str(m) for m in members]
    print(
        f"{name:18s} {sum(docs[k] for k in keys):6d} {sum(docs[k] for k in keys)/n:7.1%} "
        f"{sum(toks[k] for k in keys)/total_tokens:8.1%} {sum(capped[k] for k in keys)/total_capped:8.1%} "
        f"{sum(positions[k] for k in keys)/total_share:10.1%}"
    )
