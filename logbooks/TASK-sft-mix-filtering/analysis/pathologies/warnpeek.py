import sys, re, json, collections, pandas as pd
ag = pd.read_parquet(sys.argv[1]); model = sys.argv[2]; bench = sys.argv[3]
a = ag[(ag.model==model)&(ag.bench==bench)].copy()
# warning on step k's observation refers to step k's response
a["pre"] = a.text.map(lambda s: s[:s.find("{")] if "{" in s else s)
a["prelen"] = a.pre.str.strip().str.len()
print("reasoning_content nonempty:", (a.reasoning.str.len()>0).mean().round(3))
print("warned steps:", a.warn.mean().round(3), "| warned steps with text before first '{':", (a[a.warn].prelen>0).mean().round(3), "| unwarned with pre-text:", (a[~a.warn].prelen>0).mean().round(3))
print("pre-text length among warned: median", a[a.warn].prelen.median(), "p90", a[a.warn].prelen.quantile(.9))
kinds = collections.Counter()
for o in a.observation[a.warn|a.perr]:
    for k in re.findall(r"(Extra text detected (?:before|after) JSON object|Missing required fields?[^\\\"\n]{0,40}|No valid JSON[^\\\"\n]{0,40}|Invalid JSON[^\\\"\n]{0,40})", o): kinds[k[:50]] += 1
print(kinds.most_common(6))
for r in a[a.warn].sample(4, random_state=2).itertuples():
    print("--- step", r.step, "PRE:", repr(r.pre[:300])); print("    TAIL:", repr(r.text[-200:]))
