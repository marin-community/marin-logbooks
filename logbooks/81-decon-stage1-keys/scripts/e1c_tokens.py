# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""E1c: planted recall for tokenizer dumps, accent folding and other character-level forms."""

import unicodedata
from collections import defaultdict

exec(open("work/e1_planted.py").read().split("HOMO = ")[0])


def strip_accents(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


FORMS2 = {
    "gpt2_bytelevel": lambda t: "".join(("Ġ" + w) if i else w for i, w in enumerate(t.replace("\n", " Ċ ").split(" "))),
    "sentencepiece": lambda t: "▁" + t.replace(" ", "▁"),
    "wordpiece": lambda t: " ".join(w[:3] + " ##" + w[3:] if len(w) > 5 else w for w in t.split(" ")),
    "strip_accents": strip_accents,
    "fullwidth": lambda t: "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in t),
    "turkish_upper": lambda t: t.replace("i", "İ").replace("I", "I"),
    "nbsp_unicode": lambda t: t.replace(" ", " "),
    "smart_quotes": lambda t: t.replace("'", "’").replace('"', "“"),
    "latex_wrapped": lambda t: "\\textbf{" + t.replace("\n", "\\\\\n") + "}",
}
host = "The committee met on Tuesday to review the budget and the new library hours. " * 40
accented = [i for i in items if strip_accents(text_of(i)) != text_of(i)]
found = defaultdict(int)
for item in items:
    text = text_of(item)
    for name, form in FORMS2.items():
        record = host + "\n" + form(text) + "\n" + host
        found[name] += any(p.item == item["id"] for p in record_pairs("r", record, table, locators))
acc = sum(
    any(p.item == i["id"] for p in record_pairs("r", host + strip_accents(text_of(i)) + host, table, locators))
    for i in accented
)
for name in FORMS2:
    print(f"{name:16s} {found[name] / len(items):.3f}")
print(f"strip_accents on the {len(accented)} items with accents: {acc / max(len(accented), 1):.3f}")
