import sys, re, json, pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["source","tokens","text"])
src = pd.DataFrame(json.load(open(sys.argv[2]))["sources"]).set_index("name")
TURN = re.compile(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)(?=<\|eot_id\|>|$)", re.S)
PAT = {"nnbsp": re.compile("[ ‑]"), "cjk": re.compile(r"[一-鿿぀-ヿ]"),
       "giveup": re.compile(r"(?i)given (the |that |my )?(limited |lack of )?time|I(?:'ll| will) guess|best guess|I give up|placeholder")}
def has(p, t): return any(p.search(b) for b in TURN.findall(t))
for k, p in PAT.items(): df[k] = df.text.map(lambda t: has(p, t))
g = df.groupby("source").agg(n=("tokens","size"), **{k: (k, "mean") for k in PAT})
g = g.join(src["share"], how="left").fillna({"share": 0})
for k in PAT:
    print(f"{k}: share-weighted fraction of training documents with it in an assistant turn = {(g[k]*g.share).sum()/g.share.sum():.3f}")
print(g.sort_values("share", ascending=False).head(25).round(3).to_string())
