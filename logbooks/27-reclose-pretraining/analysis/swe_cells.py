"""Estimate swe-zero-12m's token share of each datakit (cluster, quality) cell in store_8ac06c74."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for

ASSIGN = "gs://marin-us-central2/datakit/cluster_assign/swe-zero-12m_e217479f"
QUALITY = "gs://marin-us-central2/datakit/quality/swe-zero-12m_53ff59eb"
TOKENIZE = "gs://marin-us-central2/datakit/tokenize/swe-zero-12m_5c2fa998/train"
N = 470
STRIDE = int(sys.argv[1])
THRESH = [0.2, 0.4, 0.6, 0.8]

def read(url, cols):
    fs, path = filesystem_for(url)
    with fs.open(path, "rb") as h:
        return pq.read_table(h, columns=cols).to_pandas()

def listing(url):
    fs, path = filesystem_for(url)
    return sorted("gs://" + f for f in fs.find(path) if f.endswith(".parquet"))
A_FILES, Q_FILES, T_FILES = listing(ASSIGN), listing(QUALITY), listing(TOKENIZE)
assert len(A_FILES) == len(Q_FILES) == len(T_FILES) == N, (len(A_FILES), len(Q_FILES), len(T_FILES))

def shard(i):
    a = read(A_FILES[i], ["id", "cluster_40"])
    q = read(Q_FILES[i], ["id", "score"])
    m = a.merge(q, on="id", how="inner")
    assert len(m) == len(a) == len(q), (i, len(a), len(q), len(m))
    m["qb"] = np.searchsorted(THRESH, m["score"].to_numpy(), side="right")
    return m

idx = list(range(0, N, STRIDE))
with ThreadPoolExecutor(16) as pool:
    parts = list(pool.map(shard, idx))
import pandas as pd
df = pd.concat(parts)
scale = N / len(idx)
counts = df.groupby(["cluster_40", "qb"]).size()
print("sampled docs", len(df), "scaled", int(len(df) * scale))

# token lengths from one tokenize shard, joined to its cell
i0 = idx[0]
fs, path = filesystem_for(T_FILES[i0])
with fs.open(path, "rb") as h:
    t = pq.read_table(h).to_pandas()
t["ntok"] = t["input_ids"].map(len)
t = t[["id", "ntok"]].merge(parts[0], on="id")
print("tokenize shard docs", len(t), "mean tokens/doc", t["ntok"].mean())
len_by_cell = t.groupby(["cluster_40", "qb"])["ntok"].mean()
mean_len = t["ntok"].mean()

out = {}
for (c, q), n in counts.items():
    ml = float(len_by_cell.get((c, q), mean_len))
    out[f"c{c:02d}q{q}"] = {"docs": int(n * scale), "mean_len": ml, "tokens": int(n * scale * ml)}
json.dump(out, open(sys.argv[2], "w"), indent=1)
tot = sum(v["tokens"] for v in out.values())
print("estimated swe-zero tokens total", tot, "(registry: 106.91B)")
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["tokens"])[:15]:
    print(k, v)
