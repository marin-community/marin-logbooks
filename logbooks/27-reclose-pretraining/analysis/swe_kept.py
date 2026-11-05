import json, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
ASSIGN = "gs://marin-us-central2/datakit/cluster_assign/swe-zero-12m_e217479f"
QUALITY = "gs://marin-us-central2/datakit/quality/swe-zero-12m_53ff59eb"
DEDUP = "gs://marin-us-central2/datakit/dedup_dabe67c2/outputs/source_112"
THRESH = [0.2, 0.4, 0.6, 0.8]; N = 470
def listing(url):
    fs, path = filesystem_for(url); return sorted("gs://" + f for f in fs.find(path) if f.endswith(".parquet"))
def read(url, cols=None):
    fs, path = filesystem_for(url)
    with fs.open(path, "rb") as h: return pq.read_table(h, columns=cols).to_pandas()
A, Q, Dd = listing(ASSIGN), listing(QUALITY), listing(DEDUP)
assert len(A) == len(Q) == len(Dd) == N
rows = []
for i in range(0, N, 50):
    a = read(A[i], ["id", "cluster_40"]); q = read(Q[i], ["id", "score"]); d = read(Dd[i])
    d["canon"] = d["attributes"].map(lambda x: x["is_cluster_canonical"])
    m = a.merge(q, on="id").merge(d[["id", "canon"]], on="id", how="left")
    m["kept"] = m["canon"].isna() | (m["canon"] == True)
    m["qb"] = np.searchsorted(THRESH, m["score"].to_numpy(), side="right")
    rows.append(m[["cluster_40", "qb", "kept"]])
df = pd.concat(rows)
print("docs", len(df), "in attr file", int(df.kept.eq(False).sum() + 0), "kept overall", f"{df.kept.mean():.3%}")
g = df.groupby(["cluster_40", "qb"])["kept"].agg(["mean", "size"])
old = json.load(open(sys.argv[1])); new = {}
for k, v in old.items():
    c, q = int(k[1:3]), int(k[4])
    kf = float(g["mean"].get((c, q), df.kept.mean()))
    new[k] = {**v, "kept_frac": kf, "docs": int(v["docs"] * kf), "tokens": int(v["tokens"] * kf)}
json.dump(new, open(sys.argv[2], "w"), indent=1)
for k in ("c01q3", "c01q4", "c01q2", "c01q1", "c01q0"):
    print(k, "kept", f"{new[k]['kept_frac']:.2%}", "sample n", int(g['size'].get((int(k[1:3]), int(k[4])), 0)), "-> docs", f"{new[k]['docs']:,}", "tokens", f"{new[k]['tokens']/1e9:.2f}B")
print("store swe-zero tokens ~", f"{sum(v['tokens'] for v in new.values())/1e9:.2f}B")
