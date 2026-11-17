import sys, pandas as pd, numpy as np
df = pd.read_pickle(sys.argv[1]); rng = np.random.default_rng(1)
TH = [50_000, 100_000, 150_000, 200_000, 250_000]
for bench in ["aime24","olympiadbench"]:
    print("##", bench)
    s = df[(df.bench==bench)]
    for (arm, step), g in s.groupby(["arm","step"]):
        print(f"  {arm:14s} {step}", "  ".join(f">={t//1000}k:{(g.L>=t).mean()*100:5.1f}%" for t in TH))
    c = s[(s.arm=="control")&(s.step=="s222")]
    for arm in ["cap32k","cap32k-giveup"]:
        a = s[(s.arm==arm)&(s.step=="s222")]; out=[]
        for t in TH:
            x = (a.assign(v=a.L>=t).groupby("doc").v.mean()); y = (c.assign(v=c.L>=t).groupby("doc").v.mean()); x,y = x.align(y, join="inner")
            d = (x-y).values; bs = d[rng.integers(0,len(d),(5000,len(d)))].mean(1)
            out.append(f">={t//1000}k {d.mean()*100:+.1f} [{np.quantile(bs,.025)*100:+.1f},{np.quantile(bs,.975)*100:+.1f}]")
        print(f"  paired {arm} - control:", " | ".join(out))
