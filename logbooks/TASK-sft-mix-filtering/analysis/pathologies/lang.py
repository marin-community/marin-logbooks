"""Script-based language pathologies.

Usage: python lang.py <turns.parquet>... -- <out.parquet>

- soup: a 400-character window holding four or more non-Latin scripts, or two
  or more U+FFFD characters (mixed-script gibberish).
- answer_lang_switch: English prompt (no non-Latin letters), and the last 1,500
  characters of the output are at least 30% non-Latin letters, without soup.
- code_switch: English prompt, a run of one to four CJK characters between
  Latin words, without soup or an answer-language switch.
"""

import re
import sys

import pandas as pd

CJK_RUN = re.compile(r"(?<![\u4e00-\u9fff\u3040-\u30ff])[\u4e00-\u9fff\u3040-\u30ff]{1,4}(?![\u4e00-\u9fff\u3040-\u30ff])")
LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


def code_switch(s):
    for m in CJK_RUN.finditer(s):
        if LATIN_WORD.search(s[max(0, m.start() - 12) : m.start()]) and LATIN_WORD.search(s[m.end() : m.end() + 12]):
            return True
    return False


RANGES = {
    "CJK": "\u4e00-\u9fff\u3400-\u4dbf", "KANA": "\u3040-\u30ff", "HANGUL": "\uac00-\ud7af\u1100-\u11ff",
    "CYRILLIC": "\u0400-\u04ff", "GREEK": "\u0370-\u03ff\u1f00-\u1fff", "ARABIC": "\u0600-\u06ff", "HEBREW": "\u0590-\u05ff",
    "THAI": "\u0e00-\u0e7f", "INDIC": "\u0900-\u0dff", "GEORGIAN": "\u10a0-\u10ff", "ARMENIAN": "\u0530-\u058f",
    "ETHIOPIC": "\u1200-\u137f", "SEA": "\u1000-\u109f\u1780-\u17ff", "MONGOLIAN": "\u1800-\u18af",
}
SCRIPT_RE = {k: re.compile(f"[{v}]") for k, v in RANGES.items()}
NONLATIN = re.compile("[" + "".join(RANGES.values()) + "]")


def soup(s):
    if s.count("\ufffd") >= 2:
        return True
    if len(NONLATIN.findall(s)) < 4:
        return False
    for i in range(0, len(s), 200):
        w = s[i : i + 400]
        if NONLATIN.search(w) and sum(1 for r in SCRIPT_RE.values() if r.search(w)) >= 4:
            return True
    return False


def nonlatin_letter_share(s):
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 50:
        return 0.0
    return sum(1 for c in letters if NONLATIN.match(c)) / len(letters)


def english_prompt(p):
    return nonlatin_letter_share(p) < 0.01


paths = sys.argv[1 : sys.argv.index("--")]
out = sys.argv[sys.argv.index("--") + 1]
t = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
for c in ["text", "reasoning", "tool_calls", "prompt"]:
    t[c] = t[c].fillna("")
whole = t.reasoning + "\n" + t.text + "\n" + t.tool_calls
t["soup"] = whole.map(soup)
eng = t.prompt.map(english_prompt)
t["english_prompt"] = eng
t["answer_lang_switch"] = eng & ~t.soup & t.text.map(lambda s: nonlatin_letter_share(s[-1500:]) >= 0.30)
t["code_switch"] = eng & ~t.soup & ~t.answer_lang_switch & whole.map(code_switch)
t.drop(columns=["prompt"]).to_parquet(out)
r = t.groupby(["model", "bench"])[["soup", "answer_lang_switch", "code_switch"]].mean().mul(100).round(2)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(r[(r > 0).any(axis=1)].to_string())
