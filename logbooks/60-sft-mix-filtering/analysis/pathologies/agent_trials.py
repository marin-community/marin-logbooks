"""Trial-level agent pathologies for Terminus-harness evals (TBLite, TB2) and DS-1000.

Usage: python agent_trials.py <turns.parquet> <out_steps.parquet>
"""

import re
import sys
import unicodedata

import pandas as pd

SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "THAI", "DEVANAGARI",
           "TAMIL", "TELUGU", "BENGALI", "GEORGIAN", "ARMENIAN", "MONGOLIAN", "ETHIOPIC", "KHMER", "MYANMAR", "TIBETAN")


def scripts(s):
    seen = set()
    for ch in s:
        if ord(ch) < 0x370:
            continue
        name = unicodedata.name(ch, "")
        for sc in SCRIPTS:
            if name.startswith(sc):
                seen.add(sc)
                break
    return seen


def token_soup(s):
    """Mixed-script gibberish: four or more non-Latin scripts in one step, or replacement characters."""
    return len(scripts(s)) >= 4 or s.count("�") >= 2


t = pd.read_parquet(sys.argv[1])
ag = t[t.kind == "agent"].copy()
for c in ["text", "reasoning", "tool_calls", "observation"]:
    ag[c] = ag[c].fillna("")
ag = ag.sort_values(["run", "trial", "step"])
ag["all"] = ag.text + "\n" + ag.reasoning + "\n" + ag.tool_calls
ag["warn"] = ag.observation.str.contains("Previous response had warnings")
ag["perr"] = ag.observation.str.contains("Previous response had parsing errors")
ag["soup"] = ag["all"].map(token_soup)
ag["done"] = ag.tool_calls.str.contains("task_complete") | ag.text.str.contains(r'"task_complete"\s*:\s*true')
g = ag.groupby(["run", "trial"])
ag["prev_bad"] = (g.warn.shift(1, fill_value=False) | g.perr.shift(1, fill_value=False)).astype(bool)
ag["prev_all"] = g["all"].shift(1)
ag["same_as_prev"] = ag["all"] == ag.prev_all
ag["first_soup"] = ag.soup & ~g.soup.shift(1, fill_value=False).astype(bool)
ag.to_parquet(sys.argv[2])

rows = []
for (bench, model), b in ag.groupby(["bench", "model"]):
    tr = b.groupby("trial").agg(
        passed=("correct", "first"), steps=("step", "size"), warn=("warn", "any"), soup=("soup", "any"), done=("done", "any"),
        warn_share=("warn", "mean"), soup_steps=("soup", "sum"))
    pb = b[b.prev_bad]
    rows.append(dict(
        bench=bench, model=model, trials=len(tr), pass_rate=(tr.passed.fillna(0) > 0).mean(),
        warned_steps=b.warn.mean(), parse_err_steps=b.perr.mean(),
        identical_after_bad=pb.same_as_prev.mean() if len(pb) else float("nan"),
        rebad_after_bad=(pb.warn | pb.perr).mean() if len(pb) else float("nan"),
        soup_steps=b.soup.mean(), soup_trials=tr.soup.mean(),
        pass_rate_soup=(tr[tr.soup].passed.fillna(0) > 0).mean() if tr.soup.any() else float("nan"),
        declared_done=tr.done.mean(), done_and_failed_share=((tr.done) & ~(tr.passed.fillna(0) > 0)).sum() / max(tr.done.sum(), 1),
    ))
pd.set_option("display.width", 300)
print(pd.DataFrame(rows).round(3).T.to_string())
# Where in the trial does soup begin, relative to trial length and context?
s = ag[ag.first_soup]
if len(s):
    n = ag.groupby(["run", "trial"]).step.max()
    rel = s.apply(lambda r: r.step / n[(r.run, r.trial)], axis=1)
    print("\nfirst token-soup step position within trial (fraction of steps): quartiles", rel.quantile([.25, .5, .75]).round(2).tolist())
    after = ag.merge(s[["run", "trial", "step"]].rename(columns={"step": "soup_step"}), on=["run", "trial"])
    after = after[after.step > after.soup_step]
    print("share of steps after first soup that are also soup:", round(after.soup.mean(), 3), "n=", len(after))
