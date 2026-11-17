import sys, re, pandas as pd
sys.path.insert(0, sys.argv[2])
import importlib.util
spec = importlib.util.spec_from_file_location("d", sys.argv[2] + "/patterns.py"); d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
fl = pd.read_parquet(sys.argv[1]); col = sys.argv[3]; n = int(sys.argv[4]); model = sys.argv[5] if len(sys.argv) > 5 else None
pat = d.PATS[col]
h = fl[fl[col]]
if model: h = h[h.model == model]
print(col, "hits", len(h), h.groupby("bench").size().to_dict())
for r in h.sample(min(n, len(h)), random_state=0).itertuples():
    s = r.reasoning + "\n" + r.text + "\n" + r.tool_calls
    m = pat.search(s)
    if not m: print("  (no match in reconstruct)", r.bench); continue
    print(f"--- {r.model} {r.bench} doc={str(r.doc)[:30]} correct={r.correct} len={len(s)} pos={m.start()}")
    print("   ", repr(s[max(0, m.start()-220): m.end()+180]))
