import sys, time, re, pandas as pd
src = open(sys.argv[2]).read().split("paths = sys.argv")[0]; exec(src)
t = pd.read_parquet(sys.argv[1]).sample(2000, random_state=0)
for c in ["text","reasoning","tool_calls","prompt"]: t[c] = t[c].fillna("")
w = t.reasoning + t.text + t.tool_calls
for name, f, col in [("soup", soup, w), ("eng", english_prompt, t.prompt), ("share_tail", lambda s: nonlatin_letter_share(s[-1500:]), t.text), ("cs", lambda s: bool(CODE_SWITCH.search(s)), w)]:
    t0 = time.time(); col.map(f); print(name, round(time.time()-t0, 2))
