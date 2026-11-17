import glob, os, re, pyarrow.parquet as pq, pyarrow as pa, pandas as pd
def runs(root):
    out = {}
    for d in sorted(glob.glob(f"{root}/*")):
        m = re.match(r"sft11188-(control|cap32k-giveup|cap32k)-(s\d+b?)-(.+)-[0-9a-f]{32}$", os.path.basename(d))
        if not m: continue
        fs = glob.glob(f"{d}/samples/**/*.parquet", recursive=True)
        if not fs: continue
        t = pa.concat_tables([pq.read_table(f) for f in fs], promote_options="default").to_pandas()
        t = t.sort_values("_seq").drop_duplicates(["doc_id", "trial_id"], keep="last")
        t["output"] = t.output.fillna("")
        arm, step, bench = m.groups()
        out[(arm, step.rstrip("b"), bench)] = t
    return out
