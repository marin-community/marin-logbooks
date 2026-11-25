# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""E1b: planted recall of the missed forms after one combined extra decoding."""

import sys
from collections import defaultdict

sys.argv = ["x"]
exec(open("work/e1_planted.py").read().split("host = (")[0])
from experiments.datakit.decontam.encoder.encoding_survey import (
    _base64,
    _code_escapes,
    _homoglyphs,
    _html,
    _numbered_lines,
    _url,
)


def combined(t):
    return _numbered_lines(_code_escapes(_html(_url(t))))


FIXES = {"combined": combined, "combined+homoglyph": lambda t: _homoglyphs(combined(t)), "base64": _base64}
host = "The committee met on Tuesday to review the budget and the new library hours. " * 40
targets = [
    "url_percent",
    "url_plus",
    "url_in_query",
    "html_escape",
    "html_nbsp",
    "html_numeric_punct",
    "html_tags_words",
    "html_br_lines",
    "line_numbers",
    "c_escape_x",
    "homoglyphs",
    "base64",
    "rot13",
    "quoted_printable",
]
sub = items[::3]
found = defaultdict(int)
for item in sub:
    text = text_of(item)
    for name in targets:
        record = host + "\n" + FORMS[name](text) + "\n" + host
        for fix, decode in FIXES.items():
            found[(name, fix)] += any(p.item == item["id"] for p in record_pairs("r", decode(record), table, locators))
print(len(sub), "items")
for name in targets:
    print(f"{name:20s}", "  ".join(f"{fix} {found[(name, fix)] / len(sub):.3f}" for fix in FIXES))
