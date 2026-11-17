import sys, re, collections, pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["source","doc_index","tokens","text"])
TURN = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>(.*?)(?=<\|eot_id\|>|$)", re.S)
term = df[df.text.str.contains("Format your response as JSON", regex=False)]
print("Terminus-format docs:", len(term), term.source.value_counts().to_dict())
stats = collections.Counter(); examples = []
for r in term.itertuples():
    turns = TURN.findall(r.text)
    for i, (role, body) in enumerate(turns):
        if role != "assistant": continue
        think = re.search(r"<\|start_think\|>(.*?)<\|end_think\|>", body, re.S)
        ans = body[think.end():] if think else body
        ans_s = ans.lstrip()
        stats["turns"] += 1
        stats["has_think"] += bool(think)
        stats["answer_starts_json"] += ans_s.startswith("{")
        stats["answer_prose_then_json"] += (not ans_s.startswith("{")) and ("{" in ans_s)
        stats["answer_Analysis_prefix"] += bool(re.match(r"(Analysis|ANALYSIS)\s*:", ans_s))
        stats["think_Analysis_prefix"] += bool(think and re.match(r"\s*(Analysis)\s*:", think.group(1)))
        nxt = turns[i+1][1] if i+1 < len(turns) else ""
        w = "Previous response had warnings" in nxt; stats["next_warned"] += w
        stats["next_warned_and_answer_json_clean"] += w and ans_s.startswith("{")
        if (not ans_s.startswith("{")) and len(examples) < 4: examples.append((r.source, r.doc_index, ans_s[:300]))
print(dict(stats))
for e in examples: print("  EX", e)
# overall think usage in Terminus docs vs eval: what does a think span hold?
r = term.iloc[0]; print(r.text[:200].replace("\n"," | "))
