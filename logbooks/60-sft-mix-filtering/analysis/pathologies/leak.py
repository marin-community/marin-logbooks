import sys, re, pandas as pd
t = pd.read_parquet(sys.argv[1], columns=["model","bench","doc","text","correct"])
NOTHINK = {"gsm8k-0shot","humanevalplus","ifbench","mbppplus","triviaqa","nupa","truthfulqa","financebench","mrcr","mrcr-32k","mrcr-65k"}
t = t[t.bench.isin(NOTHINK)]
OPEN = re.compile(r"^\s*(The user (is asking|asks|wants|requests|has)|We need to|We have to|We are (asked|given)|I need to|Let me (think|analyze|figure|work)|Okay, (so|let)|Hmm)", re.I)
HANDOFF = re.compile(r"(Let's|Let me|Now|We'll|I'll) (produce|craft|write|output|give|provide) (the )?final answer|Provide (the )?(final )?answer\.|Thus answer:?\s*\S+\.\s*(Provide|Let)|Let's produce final", re.I)
t["open"] = t.text.map(lambda x: bool(OPEN.search(x[:200])))
t["handoff"] = t.text.map(lambda x: bool(HANDOFF.search(x)))
t["leak"] = t.open | t.handoff
r = t.groupby(["bench","model"])[["open","handoff","leak"]].mean().mul(100).round(1).unstack("model")["leak"]
print(r.to_string())
print("\naccuracy when leak vs not (gsm8k, humaneval, ifbench):")
for b in ["gsm8k-0shot","humanevalplus","ifbench"]:
    g = t[t.bench==b].groupby(["model","leak"]).correct.mean().unstack().round(3); print(b); print(g.to_string())
t.to_parquet(sys.argv[2])
