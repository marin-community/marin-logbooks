import sys, glob, pyarrow.parquet as pq, pyarrow as pa, pandas as pd, json
d = sys.argv[1]
fs = glob.glob(f"{d}/rollouts_v1/**/*.parquet", recursive=True); print("rollout files", len(fs))
t = pa.concat_tables([pq.read_table(f) for f in fs[:40]], promote_options="default").to_pandas()
print(t.groupby(["participant_type","content_type"]).size())
a = t[t.participant_type!="user"]
for r in a.head(3).itertuples():
    ids = r.completion_token_ids
    print("--- tokens:", None if ids is None else len(ids), "think ids:", None if ids is None else (list(ids).count(128002), list(ids).count(128003)))
    print("   content head:", repr(str(r.content)[:400]))
    print("   meta:", str(r.metadata_json)[:300])
