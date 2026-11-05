import re, pyarrow.parquet as pq, pandas as pd
from transformers import AutoTokenizer
from rigging.filesystem.buckets import filesystem_for
tok = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
def listing(url):
    fs, path = filesystem_for(url); return sorted("gs://" + f for f in fs.find(path) if f.endswith(".parquet"))
T = listing("gs://marin-us-central2/datakit/tokenize/swe-zero-12m_5c2fa998/train")[0]
Dd = listing("gs://marin-us-central2/datakit/dedup_dabe67c2/outputs/source_112")[0]
def read(u):
    fs, p = filesystem_for(u)
    with fs.open(p, "rb") as h: return pq.read_table(h).to_pandas()
t = read(T); d = read(Dd)
d["canon"] = d["attributes"].map(lambda a: a["is_cluster_canonical"])
t = t.merge(d[["id", "canon"]], on="id", how="left"); t["kept"] = t["canon"].isna() | (t["canon"] == True)
ASSIST = re.compile(r"<assistant>\n(.*?)\n</assistant>", re.DOTALL)
CLOSE = re.compile(r"</think(?:ing)?>")
OPEN = re.compile(r"<think(?:ing)?>")
rows = []
for ids, kept in zip(t["input_ids"], t["kept"]):
    text = tok.decode(list(ids))
    turns = ASSIST.findall(text)
    n_close = sum(len(CLOSE.findall(x)) for x in turns)
    n_open = sum(len(OPEN.findall(x)) for x in turns)
    turns_close = sum(1 for x in turns if CLOSE.search(x))
    turns_multi = sum(1 for x in turns if len(CLOSE.findall(x)) >= 2)
    # close followed by non-whitespace prose before the bash fence
    turns_close_continue = sum(1 for x in turns if re.search(r"</think(?:ing)?>\s*\S(?:(?!```bash).)*?</think(?:ing)?>", x, re.DOTALL) or re.search(r"</think(?:ing)?>\s*[^\s`]", x))
    rows.append((kept, len(turns), n_close, n_open, turns_close, turns_multi, turns_close_continue, len(ids)))
df = pd.DataFrame(rows, columns=["kept", "turns", "closes", "opens", "turns_close", "turns_multi", "turns_close_continue", "ntok"])
for name, g in (("all", df), ("kept", df[df.kept]), ("dropped", df[~df.kept])):
    print(f"{name:8} docs={len(g):6d} turns/doc={g.turns.mean():5.1f} docs_with_close={g.closes.gt(0).mean():6.1%} docs_with_reclose_turn={g.turns_multi.gt(0).mean():6.1%} "
          f"turns_with_close={g.turns_close.sum()/g.turns.sum():6.1%} turns_with_2+close={g.turns_multi.sum()/g.turns.sum():6.2%} turns_close_then_prose={g.turns_close_continue.sum()/g.turns.sum():6.2%} opens/closes={g.opens.sum()}/{g.closes.sum()}")
