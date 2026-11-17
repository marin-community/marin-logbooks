import sys, re, zlib, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2]); from load import runs
R = runs(sys.argv[1]); OUT = sys.argv[3]
MATH_GIVE_UP = re.compile(
    r"(?i)(given (the |that |my )?(limited |amount of )?time|time (is |are )?(limited|constraints?|up)\b|out of time"
    r"|time I have spent|spent (so much|too much|a lot of|enough|considerable) time"
    r"|\bI(?:'ll| will) guess|educated guess|best guess|my guess|as a guess|output the conjecture"
    r"|I(?:'ll| will) conjecture|cannot (produce|find|determine|derive) (a |the )?(correct|exact|closed|rigorous)"
    r"|I give up|ignore the .{0,30}issue|many might accept|hope(fully)? (it|this|that)('s| is)? (correct|right|accepted)"
    r"|without (a )?(full |rigorous |complete )?proof|I think I (must|have to|should) (conclude|stop|give))")
TIME = re.compile(r"(?i)given (the |that |my )?(limited |amount of )?time|time (is |are )?(limited|constraints?|up)\b|out of time|time I have spent|spent (so much|too much|a lot of|enough|considerable) time")
GUESS = re.compile(r"(?i)\bI(?:'ll| will) guess|educated guess|best guess|my guess|as a guess|I give up")
GIST = re.compile(r"I give up|I'll (?:just )?guess|I will guess|best guess|placeholder|given the (?:lack of )?time|running out of time|cannot (?:solve|determine)|I'm not (?:sure|confident)|let's (?:just )?(?:go with|assume the answer)|I'll go with", re.I)
RESTART = re.compile(r"\b(?:Wait|Actually|Alternatively|Hmm)\b")
def exact_loop(s):
    s = s.rstrip(); n = len(s)
    if n < 600: return 0
    for p in range(1, 2001):
        if 2*p > n: break
        if s[-p:] != s[-2*p:-p]: continue
        unit = s[-p:]; k = 2
        while (k+1)*p <= n and s[-(k+1)*p:-k*p] == unit: k += 1
        if (k >= 8 and k*p >= 200) or (k >= 3 and k*p >= 2000): return k*p
    return 0
def soft_loop(s):
    if len(s) < 8000: return False
    t = s[-6000:].encode(); return len(zlib.compress(t))/len(t) < 0.10
def tail_before_box(o, n):
    i = o.rfind("\\boxed"); end = i if i >= 0 else len(o)
    return o[max(0, end-n):end]
rows = []
for (arm, step, bench), t in R.items():
    if bench not in ("aime24", "olympiadbench", "math500"): continue
    o = t.output; L = o.str.len()
    d = pd.DataFrame(dict(arm=arm, step=step, bench=bench, doc=t.doc_id.values, L=L.values,
        correct=t.correct.fillna(False).astype(bool).values))
    d["long"] = d.L >= 100_000
    d["boxed"] = o.str.contains(r"\\boxed", regex=True).values
    d["xloop"] = [exact_loop(x) > 0 for x in o]; d["sloop"] = [soft_loop(x) for x in o]; d["loop"] = d.xloop | d.sloop
    tail = [tail_before_box(x, 1500) for x in o]
    d["give_filter"] = [bool(MATH_GIVE_UP.search(x)) for x in tail]
    d["give_time"] = [bool(TIME.search(x)) for x in tail]
    d["give_guess"] = [bool(GUESS.search(x)) for x in tail]
    d["give_gist"] = [bool(GIST.search(x[-3000:])) for x in o]
    d["give_anywhere"] = [bool(TIME.search(x)) for x in o]
    d["restart_per_k"] = [len(RESTART.findall(x))/max(len(x),1)*1000 for x in o]
    d["n_boxed"] = o.str.count(r"\\boxed").values
    rows.append(d)
df = pd.concat(rows, ignore_index=True)
df.to_pickle(OUT)
print(len(df))
