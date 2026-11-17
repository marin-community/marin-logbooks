"""Flatten eval runs into one table of assistant outputs.

Usage: python turns.py <runs_root> <out.parquet>

Text benchmarks contribute one row per sample (`output`, reasoning and answer
concatenated). Agent benchmarks with a `steps` table contribute one row per
agent step (`message`, `reasoning_content`, `tool_calls_json`) plus the
observation that followed, so detectors can see the harness response.
"""

import glob
import json
import os
import re
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read(d, tbl, cols=None):
    fs = glob.glob(f"{d}/{tbl}/**/*.parquet", recursive=True)
    if not fs:
        return None
    ts = []
    for f in fs:
        t = pq.read_table(f)
        if cols:
            t = t.select([c for c in cols if c in t.column_names])
        ts.append(t)
    return pa.concat_tables(ts, promote_options="default").to_pandas()


def bench_of(name):
    m = re.search(r"2026\.10\.05-(?:[0-9a-f]{32}-)?(.+?)-[0-9a-f]{32}$", name)
    if m:
        return m.group(1)
    m = re.match(r"sft11188-(?:control|cap32k-giveup|cap32k)-s\d+b?-(.+)-[0-9a-f]{32}$", name)
    if m:
        return m.group(1)
    m = re.match(r"lce40-guards-.+?-full1-(.+)-[0-9a-f]{32}$", name)
    return m.group(1) if m else name


def model_of(name):
    m = re.match(r"sft11188-(control|cap32k-giveup|cap32k)-(s\d+)b?-", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"lce40-guards-(control-seed1|control|filtered)-step238-", name)
    return f"lce40-{m.group(1)}-s238" if m else "sft-10.05"


def prompt_text(row):
    msgs = row.get("prompt_messages")
    if msgs is None:
        return row.get("prompt_text") or ""
    return "\n".join(f"[{m['role']}] {m['content']}" for m in msgs)


rows = []
for d in sorted(glob.glob(f"{sys.argv[1]}/*")):
    name = os.path.basename(d)
    bench, model = bench_of(name), model_of(name)
    steps = read(d, "steps")
    samples = read(d, "samples")
    if steps is not None and len(steps):
        steps = steps.sort_values("_seq").drop_duplicates(["trial_id", "step_id"], keep="last")
        steps = steps.sort_values(["trial_id", "step_id"])
        reward = {}
        if samples is not None:
            for _, s in samples.sort_values("_seq").drop_duplicates("trial_id", keep="last").iterrows():
                g = s["grading"] or {}
                reward[s["trial_id"]] = g.get("score")
        for trial, g in steps.groupby("trial_id"):
            g = g.reset_index(drop=True)
            first_user = next((m for m in g.message[g.source == "user"] if isinstance(m, str)), "")
            for i, s in g.iterrows():
                if s["source"] != "agent":
                    continue
                nxt = s["observation_json"] or ""
                rows.append(
                    dict(model=model, bench=bench, kind="agent", doc=s["doc_id"], trial=trial, step=int(s["step_id"]),
                         text=s["message"] or "", reasoning=s["reasoning_content"] or "",
                         tool_calls=s["tool_calls_json"] or "", observation=nxt, prompt=first_user[:20000],
                         correct=reward.get(trial), run=name)
                )
        continue
    if samples is None or not len(samples) or "output" not in samples:
        print("skip (no text outputs)", name, file=sys.stderr)
        continue
    samples = samples.sort_values("_seq").drop_duplicates(["doc_id", "trial_id"], keep="last")
    if samples.output.isna().all():
        print("skip (null outputs)", name, file=sys.stderr)
        continue
    for _, s in samples.iterrows():
        rows.append(
            dict(model=model, bench=bench, kind="text", doc=s["doc_id"], trial=str(s["trial_id"]), step=0,
                 text=s["output"] or "", reasoning="", tool_calls="", observation="", prompt=prompt_text(s)[:20000],
                 correct=None if s["correct"] is None else float(bool(s["correct"])), run=name)
        )

df = pd.DataFrame(rows)
df.to_parquet(sys.argv[2])
print(df.groupby(["model", "bench", "kind"]).size().to_string())
