import sys, glob, unicodedata, pyarrow.parquet as pq, pyarrow as pa, pandas as pd, numpy as np
sys.path.insert(0, sys.argv[1]); 
from agent_trials_lib import token_soup
for d in sys.argv[2:]:
    fs = glob.glob(f"{d}/steps/**/*.parquet", recursive=True)
    st = pa.concat_tables([pq.read_table(f, columns=["trial_id","step_id","source","message","reasoning_content","tool_calls_json","prompt_tokens","completion_tokens","_seq"]) for f in fs]).to_pandas()
    st = st.sort_values("_seq").drop_duplicates(["trial_id","step_id"], keep="last")
    ag = st[st.source=="agent"].copy()
    ag["all"] = ag.message.fillna("") + ag.tool_calls_json.fillna("")
    ag["soup"] = ag["all"].map(token_soup)
    ag["pt_bin"] = pd.cut(ag.prompt_tokens, [0, 4000, 8000, 16000, 24000, 28000, 32000, 40000, 1e9])
    print("==", d.split("-")[-2])
    print(ag.groupby("pt_bin", observed=True).soup.agg(["mean","size"]).round(4).to_string())
    print("completion tokens soup vs not (median):", ag[ag.soup].completion_tokens.median(), ag[~ag.soup].completion_tokens.median())
