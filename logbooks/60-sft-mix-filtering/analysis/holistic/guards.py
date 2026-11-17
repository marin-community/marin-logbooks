import sys, json, re, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2]); from load import runs
R = runs(sys.argv[1]); rng = np.random.default_rng(0)
def metrics(t):
    return pd.DataFrame([dict(m) if m is not None else {} for m in t.metrics], index=t.doc_id.values)
for bench in ["humanevalplus","ifbench","gsm8k-0shot"]:
    print(f"\n######## {bench}")
    base = None
    for arm, step in [("control","s222"),("cap32k","s222"),("cap32k-giveup","s222"),("control","s100"),("cap32k-giveup","s100")]:
        t = R[(arm, step, bench)].set_index("doc_id"); L = t.output.str.len()
        m = metrics(t.reset_index()); 
        ok = t.correct.fillna(False).astype(bool)
        has_code = t.output.str.contains("```").mean() if bench=="humanevalplus" else np.nan
        think_leak = t.output.str.contains(r"(?i)\b(?:wait|let me think|let's think|hmm)\b").mean()
        print(f"{arm:14s} {step} n={len(t)} acc={ok.mean():.3f} medL={L.median():.0f} p95={L.quantile(.95):.0f} n>10k={int((L>10_000).sum())} acc|L>10k={ok[L>10_000].mean() if (L>10_000).any() else float('nan'):.2f} wait-words={think_leak:.3f} codeblock={has_code:.3f}")
        print("     metric means:", {k: round(v,3) for k,v in m.mean().items()})
        if arm=="control" and step=="s222": base = (ok, L, t)
        elif step=="s222":
            b_ok, b_L, bt = base; a, c = ok.align(b_ok, join="inner")
            gained = int((a & ~c).sum()); lost = int((~a & c).sum())
            from math import comb
            n = gained+lost; p = sum(comb(n,k) for k in range(0, min(gained,lost)+1))/2**n*2 if n else 1
            print(f"     vs control: gained {gained} lost {lost} exact-McNemar p={min(p,1):.3f}")
            if bench=="humanevalplus":
                lostids = a.index[~a & c]
                LL = L.reindex(lostids); print("     lost items: arm output length median", LL.median(), "control", b_L.reindex(lostids).median(), "; lost with arm L>5k:", int((LL>5000).sum()))
