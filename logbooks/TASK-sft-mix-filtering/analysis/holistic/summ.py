import sys, numpy as np, pandas as pd
df = pd.read_pickle(sys.argv[1]); rng = np.random.default_rng(0)
M = ["correct","long","boxed","loop","xloop","give_filter","give_time","give_guess","give_gist","give_anywhere"]
df["nobox"] = ~df.boxed; df["give_filter_wrong"] = df.give_filter & df.boxed & ~df.correct; df["gist_boxed_wrong"] = df.give_gist & df.boxed & ~df.correct
df["long_loop"] = df.long & df.loop
M = ["correct","long","nobox","loop","give_filter","give_filter_wrong","give_time","give_guess","gist_boxed_wrong","give_anywhere"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for bench in ["aime24","olympiadbench","math500"]:
    b = df[df.bench==bench]
    g = b.groupby(["arm","step"])
    t = g[M].mean().mul(100).round(1)
    t["n"] = g.size(); t["med_L"] = g.L.median().round(-2); t["p90_L"] = g.L.quantile(.9).round(-3)
    t["right_med"] = b[b.correct].groupby(["arm","step"]).L.median().round(-2); t["wrong_med"] = b[~b.correct].groupby(["arm","step"]).L.median().round(-2)
    t["acc_short"] = (b[~b.long].groupby(["arm","step"]).correct.mean()*100).round(1)
    t["acc_long"] = (b[b.long].groupby(["arm","step"]).correct.mean()*100).round(1)
    t["restart/k"] = g.restart_per_k.median().round(2)
    print(f"\n######## {bench}"); print(t.to_string())
# paired per-problem bootstrap at s222 (and giveup s100 vs control s100)
def paired(bench, a, sa, c="control", sc=None, metrics=M+["L"]):
    sc = sc or sa
    x = df[(df.bench==bench)&(df.arm==a)&(df.step==sa)].groupby("doc")[metrics].mean()
    y = df[(df.bench==bench)&(df.arm==c)&(df.step==sc)].groupby("doc")[metrics].mean()
    x, y = x.align(y, join="inner", axis=0); d = (x - y)
    idx = rng.integers(0, len(d), (5000, len(d)))
    out = {}
    for m in metrics:
        v = d[m].values; bs = v[idx].mean(1); s = 100 if m!="L" else 1
        out[m] = f"{v.mean()*s:+.1f} [{np.quantile(bs,.025)*s:+.1f},{np.quantile(bs,.975)*s:+.1f}]"
    return len(d), out
print("\n######## paired per-problem differences vs control, 95% bootstrap over problems")
for bench in ["aime24","olympiadbench","math500"]:
    for a, s in [("cap32k","s222"),("cap32k-giveup","s222"),("cap32k-giveup","s100")]:
        n, o = paired(bench, a, s); print(bench, a, s, f"problems={n}")
        for k,v in o.items(): print("   ", k, v)
