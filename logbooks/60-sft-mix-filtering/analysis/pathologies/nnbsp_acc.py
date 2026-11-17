import sys, re, glob, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
N = re.compile("[ ‑]")
rows = []
for root in sys.argv[1:]:
    for d in glob.glob(f"{root}/*"):
        fs = glob.glob(f"{d}/samples/**/*.parquet", recursive=True)
        if not fs: continue
        try: t = pa.concat_tables([pq.read_table(f, columns=["doc_id","trial_id","output","extracted","target_text","correct","_seq"]) for f in fs], promote_options="default").to_pandas()
        except Exception: continue
        if t.output.isna().all(): continue
        t = t.sort_values("_seq").drop_duplicates(["doc_id","trial_id"], keep="last")
        name = d.split("/")[-1]
        m = re.search(r"(aime24|olympiadbench|math500|gsm8k-0shot|ifbench|humanevalplus|mmlu-pro|gpqa-diamond|cruxeval|mbppplus|nupa|triviaqa)", name)
        if not m: continue
        t["bench"] = m.group(1); t["model"] = "10.05" if "2026.10.05" in name else re.sub(r"-(aime24|olympiadbench|math500|gsm8k-0shot|ifbench|humanevalplus).*", "", name.replace("sft11188-",""))
        rows.append(t)
df = pd.concat(rows, ignore_index=True); df["output"] = df.output.fillna(""); df["extracted"] = df.extracted.fillna("").astype(str)
df["n_out"] = df.output.map(lambda s: bool(N.search(s))); df["n_ext"] = df.extracted.map(lambda s: bool(N.search(s)))
df["ok"] = df.correct.fillna(False).astype(bool)
s = df.groupby("bench").apply(lambda g: pd.Series(dict(n=len(g), nnbsp_in_output=g.n_out.mean(), nnbsp_in_extracted=int(g.n_ext.sum()), acc_with=g.ok[g.n_out].mean(), acc_without=g.ok[~g.n_out].mean())))
print(s.round(3).to_string())
e = df[df.n_ext]
print("\nextracted answers containing U+202F/U+2011:", len(e), " wrong:", int((~e.ok).sum()))
for r in e[~e.ok].head(12).itertuples(): print(f"  [{r.bench}/{r.model}] extracted={r.extracted[:60]!r} target={str(r.target_text)[:40]!r}")
# would normalizing fix them?
fix = e[~e.ok].apply(lambda r: N.sub(lambda m: " " if m.group()==" " else "-", r.extracted).replace(" ","").strip() == str(r.target_text).replace(" ","").strip(), axis=1)
print("wrong extracted answers that match the target after removing U+202F:", int(fix.sum()) if len(fix) else 0)
print("\nmath/gsm8k extracted answers with U+202F/U+2011:")
for r in e[e.bench.isin(["math500","gsm8k-0shot"])].itertuples(): print(f"  [{r.bench}/{r.model}] ok={r.ok} extracted={r.extracted[:60]!r} target={str(r.target_text)[:40]!r}")
