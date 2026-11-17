"""Candidate pathology detectors, each tied to a data flaw in marin#11188.

Usage: python detect.py <turns.parquet> <flags.parquet>

Every detector is a regex or simple structural check over one assistant output
(text benchmarks) or one agent step (agent benchmarks). Names carry the
#11188 finding number they target. Trial-level agent detectors are computed in
`agent_trials.py`.
"""

import re
import sys
import zlib

import pandas as pd

F = re.IGNORECASE

# F1: reasoning gives up, answer hides it. Training filter's math rule plus the
# science and code phrasing from the review.
GIVE_UP = re.compile(
    r"given (the |that |my )?(limited |amount of |lack of )?time|time (is |are )?(limited|constraints?|up)\b|out of time"
    r"|time I have spent|spent (so much|too much|a lot of|enough|considerable) time"
    r"|\bI(?:'ll| will) (just )?guess|educated guess|best guess|my guess|as a guess|final answer guess"
    r"|I give up|ignore the .{0,30}issue|many might accept|hope(fully)? (it|this|that)('s| is)? (correct|right|accepted)"
    r"|I think I (must|have to|should) (conclude|stop|give)|I'm not confident|not (entirely |fully )?sure,? but",
    F,
)
CODE_GIVE_UP = re.compile(
    r"placeholder|dummy (answer|output|solution|implementation)|does NOT compute|not a (proper|correct|complete) solution"
    r"|unable to (provide|produce|solve)|may (TLE|time out|not pass)|hard-?coded? (the )?(sample|example|output)"
    r"|naive solution.{0,80}(may not|won't|will not)|user may not mind",
    F,
)
# F5: reasoning quotes an instruction the prompt does not contain.
PHANTOM = {
    "phantom_stdin_buffer": (re.compile(r"sys\.stdin\.buffer|StringIO", F), re.compile(r"sys\.stdin\.buffer|StringIO", F)),
    "phantom_cpp17": (re.compile(r"(instruction|asked|says|required|use)\W{0,20}.{0,40}C\+\+17", F), re.compile(r"C\+\+17", F)),
    "phantom_boxed_instruction": (
        re.compile(r"put (your|the) final answer within \\?\\boxed", F),
        re.compile(r"final answer within \\?\\boxed", F),
    ),
    "phantom_reference_answer": (
        re.compile(r"(provided|given|reference) (good |gold )?(response|answer|solution)\b.{0,60}(match|similar|consistent|says)|matches the provided", F),
        re.compile(r"reference (answer|response)|good response", F),
    ),
    "phantom_instructions_say": (
        re.compile(r"\bthe (instructions?|system prompt|guidelines?) (says?|said|states?|tell|require)", F),
        None,
    ),
}
# F4: multilingual v1 damage.
CJK = re.compile(r"[一-鿿぀-ヿ]")
TRANSLATOR = re.compile(
    r"(you want me to translate|translat(e|ion) (the|this) (text|explanation|following)|here is the translation|I(?:'ll| will) translate)",
    F,
)
# F6/F7/F10: agent habits.
OPENER = re.compile(r"^\W{0,3}(Perfect|Excellent|Great|Good|Awesome|Wonderful)[!.]", F)
EXCUSE = re.compile(r"(pre-?existing|unrelated (to (my|the|our) (change|fix|task))?|not related to (my|the|our) (change|fix))\b.{0,80}(fail|error|issue)|(fail|error)\w*.{0,80}(pre-?existing|unrelated to)", F)
HIDDEN = re.compile(r"hidden tests?|SWE-?bench|the grader|the verifier (will|checks|extracts)|test patch", F)
RECALL = re.compile(r"recall (the )?(actual|upstream|original) (implementation|fix|patch)|from (that|the upstream) PR", F)
SUMMARY_FILE = re.compile(r"\b[A-Z_]*(SUMMARY|CHECKLIST|EXPLANATION|VERIFICATION|CHANGES)[A-Z_]*\.md\b")
EMOJI = re.compile(r"[✅✔\U0001F389\U0001F680✨\U0001F44D]")
# F9/F11: template and harness artifacts.
TOOL_USE_LITERAL = re.compile(r"^\s*\(tool use\)\s*$", re.M)
ROLE_TAG = re.compile(r"<\|(start_header_id|end_header_id|eot_id|im_start|im_end|begin_of_text)\|>|<(system|user|assistant|tool)>\n|\[INST\]")
TOOLCALL_TEXT = re.compile(r"<tool_call>|</tool_call>|<function=|\"name\":\s*\"(execute_bash|str_replace_editor|bash)\"")
THINK_TEXT = re.compile(r"<\|?/?(start_think|end_think|think)\|?>|</?think>")
REASONING_LINE = re.compile(r"^Reasoning: /(no)?think", re.M)
ANSWER_TXT = re.compile(r"/app/answer\.txt")
MODEL_LEAK = re.compile(r"powered by the model named|Qwen3-Coder-480B|Today's date: Mon Jul 13 2026", F)
# F10: teacher typography and closing rituals.
NNBSP = re.compile("[ ‑]")
READY = re.compile(r"(Now I('ll| will) (produce|write|output)[^\n]{0,80}\n?){2,}|(I'm ready[^\n]{0,40}\n?){2,}", F)
CHAR_COUNT = re.compile(r"(?:\b\w\(\d+\)){6,}")
# F12: identity and safety labels.
IDENTITY = re.compile(r"Marin Community", F)
REFUSAL_LABEL = re.compile(r"\*\*Refusal Response:?\*\*", F)
# F2/F3: replay junk echoed in outputs.
JUNK = re.compile(r"This trajectory (failed|succeeded)|# Repository Context|_reconstruct\np0|[A-Za-z0-9+/]{400,}={0,2}")


def exact_loop(s):
    s = s.rstrip()
    n = len(s)
    if n < 600:
        return False
    for p in range(1, 2001):
        if 2 * p > n:
            break
        if s[-p:] != s[-2 * p : -p]:
            continue
        unit, k = s[-p:], 2
        while (k + 1) * p <= n and s[-(k + 1) * p : -k * p] == unit:
            k += 1
        if (k >= 8 and k * p >= 200) or (k >= 3 and k * p >= 2000):
            return True
    return False


def soft_loop(s):
    if len(s) < 8000:
        return False
    t = s[-6000:].encode()
    return len(zlib.compress(t)) / len(t) < 0.10


def tail_before_box(o, n):
    i = o.rfind("\\boxed")
    end = i if i >= 0 else len(o)
    return o[max(0, end - n) : end]


def flags(r):
    text, reasoning, prompt = r.text, r.reasoning, r.prompt
    allout = reasoning + "\n" + text + "\n" + r.tool_calls
    out = {}
    tail = tail_before_box(reasoning + "\n" + text, 3000)
    out["F1_give_up_tail"] = bool(GIVE_UP.search(tail))
    out["F1_code_give_up"] = bool(CODE_GIVE_UP.search(allout))
    for name, (pat, prompt_pat) in PHANTOM.items():
        hit = bool(pat.search(allout))
        if hit and prompt_pat is not None and prompt_pat.search(prompt):
            hit = False
        out[f"F5_{name}"] = hit
    out["F4_cjk_leak"] = bool(CJK.search(allout)) and not CJK.search(prompt)
    out["F4_translator_meta"] = bool(TRANSLATOR.search(allout)) and not re.search(r"translat", prompt, F)
    out["F10_perfect_opener"] = bool(OPENER.search(text.lstrip())) or bool(OPENER.search(reasoning.lstrip()))
    out["F6_excuse_failure"] = bool(EXCUSE.search(allout))
    out["F6_success_emoji"] = bool(EMOJI.search(text))
    out["F7_hidden_tests"] = bool(HIDDEN.search(allout)) and not HIDDEN.search(prompt)
    out["F7_recall_upstream"] = bool(RECALL.search(allout))
    out["F10_summary_file"] = bool(SUMMARY_FILE.search(allout))
    out["F6_tool_use_literal"] = bool(TOOL_USE_LITERAL.search(text))
    out["F9_role_tag"] = bool(ROLE_TAG.search(allout))
    out["F9_toolcall_in_text"] = r.kind == "text" and bool(TOOLCALL_TEXT.search(text))
    out["F9_think_marker_text"] = bool(THINK_TEXT.search(text))
    out["F9_reasoning_line"] = bool(REASONING_LINE.search(allout))
    out["F11_answer_txt"] = bool(ANSWER_TXT.search(allout)) and not ANSWER_TXT.search(prompt)
    out["F11_model_leak"] = bool(MODEL_LEAK.search(allout))
    out["F10_nnbsp"] = bool(NNBSP.search(allout)) and not NNBSP.search(prompt)
    out["F10_ready_ritual"] = bool(READY.search(allout))
    out["F10_char_count"] = bool(CHAR_COUNT.search(allout))
    out["F12_identity"] = bool(IDENTITY.search(allout)) and not IDENTITY.search(prompt)
    out["F12_refusal_label"] = bool(REFUSAL_LABEL.search(allout))
    out["F2_junk_echo"] = bool(JUNK.search(allout)) and not JUNK.search(prompt)
    whole = reasoning + text
    out["loop"] = exact_loop(whole) or soft_loop(whole)
    out["len"] = len(whole)
    return out


df = pd.read_parquet(sys.argv[1])
for c in ["text", "reasoning", "tool_calls", "observation", "prompt"]:
    df[c] = df[c].fillna("")
fl = pd.DataFrame([flags(r) for r in df.itertuples()], index=df.index)
out = pd.concat([df.drop(columns=["prompt"]), fl], axis=1)
out.to_parquet(sys.argv[2])
cols = [c for c in fl.columns if c != "len"]
rate = out.groupby(["model", "bench"])[cols].mean().mul(100)
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 60)
print(rate.loc[:, (rate > 0).any()].round(2).T.to_string())
