import sys, re, pandas as pd, numpy as np
ERR = re.compile(r"Traceback \(most recent call last\)|\bError:|\bFAILED\b|\d+ failed\b|No such file or directory|command not found|AssertionError|SyntaxError|ModuleNotFoundError|exit (code|status) [1-9]", re.I)
rng = np.random.default_rng(0)
for f in sys.argv[1:]:
    ag = pd.read_parquet(f)
    for (bench, model), b in ag.groupby(["bench","model"]):
        b = b.sort_values(["trial","step"]); rows = []
        for trial, g in b.groupby("trial"):
            g = g.reset_index(drop=True); d = g.index[g.done]
            fd = False
            if len(d) and d[0] > 0: fd = bool(ERR.search(g.observation.iloc[d[0]-1][-3000:]))
            rows.append(dict(task=g.doc.iloc[0], passed=(g.correct.iloc[0] or 0) > 0, soup=bool(g.soup.any()), falsedone=fd, steps=len(g)))
        t = pd.DataFrame(rows)
        if len(t) < 100: continue
        print(f"== {bench} / {model}: trials={len(t)} tasks={t.task.nunique()} pass={t.passed.mean():.3f}")
        for flag in ["soup","falsedone"]:
            k = t[flag]
            if k.sum() == 0: continue
            naive = k.mean() * (t.passed[~k].mean() - t.passed[k].mean())
            # within-task: compare flagged vs unflagged trials of the same task
            mixed = t.groupby("task").filter(lambda g: g[flag].any() and (~g[flag]).any())
            if len(mixed):
                per = mixed.groupby("task").apply(lambda g: g.passed[~g[flag]].mean() - g.passed[g[flag]].mean())
                wt = k.mean() * per.mean()
                bs = [k.mean()*per.sample(len(per), replace=True, random_state=i).mean() for i in range(2000)]
                wt_s = f"{wt*100:+.1f} pts [{np.quantile(bs,.025)*100:+.1f},{np.quantile(bs,.975)*100:+.1f}] from {len(per)} tasks with both"
            else: wt_s = "no task has both"
            print(f"   {flag:9s} share={k.mean():.1%}  pass flagged={t.passed[k].mean():.2f} unflagged={t.passed[~k].mean():.2f}  upper-bound lift={naive*100:+.1f} pts | within-task lift={wt_s}")
