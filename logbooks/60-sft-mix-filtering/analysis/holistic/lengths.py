import sys, numpy as np
sys.path.insert(0, sys.argv[2]); from load import runs
R = runs(sys.argv[1])
for k, t in sorted(R.items()):
    L = t.output.str.len()
    print(k, len(t), t.doc_id.nunique(), "max", L.max(), "q", [int(L.quantile(q)) for q in (.5,.9,.95,.99)], "n>=200k", int((L>=200_000).sum()), "n>=230k", int((L>=230_000).sum()))
