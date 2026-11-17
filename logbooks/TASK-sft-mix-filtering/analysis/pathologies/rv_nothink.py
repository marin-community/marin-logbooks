import sys, re, collections, pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["source","doc_index","text"])
TURN = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>(.*?)(?=<\|eot_id\|>|$)", re.S)
OPEN = re.compile(r"^\s*(The user (is asking|asks|wants|requests|has)|We need to|We have to|We are (asked|given)|I need to|Let me (think|analyze|figure|work)|Okay, (so|let)|Hmm)", re.I)
HANDOFF = re.compile(r"(Let's|Let me|Now|We'll|I'll) (produce|craft|write|output|give|provide) (the )?final answer|Let's produce final", re.I)
NNBSP = re.compile("[ ‑]")
rows = []
for r in df.itertuples():
    nothink = "Reasoning: /nothink" in r.text[:400]
    think_doc = "Reasoning: /think" in r.text[:400]
    for role, body in TURN.findall(r.text):
        if role != "assistant": continue
        has_span = "<|start_think|>" in body
        ans = body.split("<|end_think|>")[-1]
        rows.append(dict(source=r.source, nothink=nothink, think_doc=think_doc, has_span=has_span,
            leak=bool(OPEN.search(ans[:200])) or bool(HANDOFF.search(ans)), nnbsp_ans=bool(NNBSP.search(ans)), nnbsp_any=bool(NNBSP.search(body))))
t = pd.DataFrame(rows)
t["mode"] = t.apply(lambda x: "nothink" if x.nothink else ("think+span" if x.has_span else ("think-nospan" if x.think_doc else "other")), axis=1)
print(t.groupby("mode")[["leak","nnbsp_ans","nnbsp_any"]].agg(["mean","sum"]).round(3).to_string())
print("\nsources of leaked-scratchpad answers in nothink / think-nospan turns:")
print(t[t.leak & t["mode"].isin(["nothink","think-nospan"])].source.value_counts().head(12).to_string())
print("\nsources with U+202F/U+2011 in answers (top):")
s = t.groupby("source").nnbsp_ans.agg(["mean","sum","size"]); print(s[s["sum"]>0].sort_values("sum", ascending=False).head(15).round(3).to_string())
