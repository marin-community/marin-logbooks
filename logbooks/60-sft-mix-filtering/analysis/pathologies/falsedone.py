import sys, re, pandas as pd
ERR = re.compile(r"Traceback \(most recent call last\)|\bError:|\bFAILED\b|\d+ failed\b|No such file or directory|command not found|AssertionError|SyntaxError|ModuleNotFoundError|exit (code|status) [1-9]", re.I)
for f in sys.argv[1:]:
    ag = pd.read_parquet(f)
    for (bench, model), b in ag.groupby(["bench","model"]):
        b = b.sort_values(["trial","step"]); rows = []
        for trial, g in b.groupby("trial"):
            g = g.reset_index(drop=True)
            d = g.index[g.done]
            if not len(d): continue
            k = d[0]
            seen = g.observation.iloc[k-1] if k > 0 else ""   # output the agent saw before declaring
            # harness also echoes last terminal state into the declaring step's own observation; use the prior step's
            rows.append(dict(trial=trial, passed=(g.correct.iloc[0] or 0) > 0, err_before=bool(ERR.search(seen[-3000:]))))
        r = pd.DataFrame(rows)
        if not len(r): continue
        fd = r[r.err_before]
        print(f"{bench:15s} {model:26s} declared={len(r):4d}  declared-with-error-just-before={len(fd):4d} ({len(fd)/len(r):.0%})  pass|err={fd.passed.mean():.2f}  pass|clean={r[~r.err_before].passed.mean():.2f}")
