"""Typography incidence and accuracy for the utf-normalize arm against the control.

Usage: python readout.py <runs_root>

Reads `<runs_root>/sft11188-<group>-<bench>-<hash>/samples` for the groups below.
Per benchmark and group: accuracy, the share of outputs containing U+202F or
U+2011 (prompts never contain them on these benchmarks; checked), occurrences
per 10k output characters, and on AIME24/OlympiadBench the long-output
(>= 100k chars) and no-boxed shares from the primary readout. A difference is
flagged when |d| > 1.96 x the combined binomial standard error; repeated
samples of one prompt are treated as independent, so the flag is optimistic on
AIME24 and OlympiadBench (10 samples per prompt).
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PAIRS = [("control-s100b", "utf-normalize-s100"), ("control-s222", "utf-normalize-s222")]
NAME = re.compile(r"sft11188-(control-s100b|control-s222|utf-normalize-s100|utf-normalize-s222)-(.+)-[0-9a-f]{32}$")
NNBSP, NBHYPHEN = " ", "‑"
LONG_CHARS = 100_000


def load(root: str) -> pd.DataFrame:
    frames = []
    for d in sorted(glob.glob(f"{root}/*")):
        m = NAME.match(os.path.basename(d))
        if not m:
            continue
        files = glob.glob(f"{d}/samples/**/*.parquet", recursive=True)
        if not files:
            continue
        t = pa.concat_tables([pq.read_table(f) for f in files], promote_options="default").to_pandas()
        t = t.sort_values("_seq").drop_duplicates(["doc_id", "trial_id"], keep="last")
        prompts = t.prompt_messages.map(lambda ms: "" if ms is None else "\n".join(x["content"] or "" for x in ms))
        if prompts.str.contains(f"[{NNBSP}{NBHYPHEN}]").any():
            raise ValueError(f"{d}: a prompt contains a target character")
        t = pd.DataFrame(
            dict(group=m.group(1), bench=m.group(2), output=t.output.fillna(""), correct=t.correct.fillna(False).astype(bool))
        )
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def metrics(g: pd.DataFrame) -> dict:
    out = g.output
    chars = out.str.len().sum()
    row = dict(
        n=len(g),
        acc=g.correct.mean(),
        typo=out.str.contains(f"[{NNBSP}{NBHYPHEN}]").mean(),
        nnbsp=out.str.contains(NNBSP).mean(),
        nbhyphen=out.str.contains(NBHYPHEN).mean(),
        per10k=(out.str.count(NNBSP).sum() + out.str.count(NBHYPHEN).sum()) / chars * 1e4,
    )
    if g.bench.iloc[0] in ("aime24", "olympiadbench"):
        row["long"] = (out.str.len() >= LONG_CHARS).mean()
        row["nobox"] = (~out.str.contains(r"\\boxed", regex=True)).mean()
    return row


def flag(a: float, b: float, na: int, nb: int) -> str:
    se = np.sqrt(a * (1 - a) / na + b * (1 - b) / nb)
    return "*" if se > 0 and abs(b - a) > 1.96 * se else ""


def main() -> None:
    df = load(sys.argv[1])
    table = {(gr, b): metrics(g) for (gr, b), g in df.groupby(["group", "bench"])}
    cols = ["acc", "typo", "nnbsp", "nbhyphen", "long", "nobox"]
    for base, arm in PAIRS:
        print(f"\n## {arm} vs {base}  (* = |d| > 1.96 SE)")
        print(f"{'bench':15s} {'metric':9s} {base:>15s} {arm:>19s} {'diff':>8s}")
        for bench in sorted({b for _, b in table}):
            if (base, bench) not in table or (arm, bench) not in table:
                print(f"{bench:15s} missing")
                continue
            x, y = table[(base, bench)], table[(arm, bench)]
            for c in cols:
                if c not in x:
                    continue
                print(f"{bench:15s} {c:9s} {x[c]*100:14.1f}% {y[c]*100:18.1f}% {(y[c]-x[c])*100:+7.1f}{flag(x[c], y[c], x['n'], y['n'])}")
            print(f"{bench:15s} {'per10k':9s} {x['per10k']:15.2f} {y['per10k']:19.2f}")


if __name__ == "__main__":
    main()
