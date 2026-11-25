# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""E1: plant real items under the production v11 keys in encodings the readings may not undo."""

import base64
import codecs
import html
import json
import quopri
import random
import re
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from marin.datakit.decon_keys import KeyTable, record_pairs

S1 = Path("work")
locators = pq.read_table(S1 / "locators.parquet")
table = KeyTable(locators.column("key1_hash").to_numpy(), np.arange(locators.num_rows))
keyed = set(locators.column("item").to_pylist())
rng = random.Random(0)
items = []
for path in sorted((S1 / "items").glob("*.parquet")):
    rows = [
        r for r in pq.read_table(path, columns=["id", "eval", "question", "choices"]).to_pylist() if r["id"] in keyed
    ]
    items += rng.sample(rows, min(8, len(rows)))
print(len(items), "items from", len({i["eval"] for i in items}), "evals", file=sys.stderr)


def text_of(item):
    choices = item["choices"] or []
    return item["question"] + (
        "\n" + "\n".join(f"{'ABCDEFGHIJ'[k]}. {c}" for k, c in enumerate(choices[:10])) if choices else ""
    )


HOMO = str.maketrans({"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х"})
FORMS = {
    "plain": lambda t: t,
    "url_percent": lambda t: urllib.parse.quote(t),
    "url_plus": lambda t: urllib.parse.quote_plus(t),
    "url_in_query": lambda t: "https://example.com/search?q=" + urllib.parse.quote_plus(t) + "&lang=en",
    "html_escape": lambda t: html.escape(t),
    "html_nbsp": lambda t: t.replace(" ", "&nbsp;"),
    "html_numeric_punct": lambda t: re.sub(r"[^\w\s]", lambda m: f"&#{ord(m.group(0))};", t),
    "html_tags_words": lambda t: " ".join(f"<span>{w}</span>" for w in t.split(" ")),
    "html_br_lines": lambda t: t.replace("\n", "<br/>"),
    "quoted_printable": lambda t: quopri.encodestring(t.encode()).decode(),
    "base64": lambda t: base64.b64encode(t.encode()).decode(),
    "json_ascii": lambda t: json.dumps({"q": t}, ensure_ascii=True),
    "json_twice": lambda t: json.dumps({"q": json.dumps({"q": t})}),
    "python_repr_bytes": lambda t: repr(t.encode()),
    "csv_quoted": lambda t: '"' + t.replace('"', '""') + '"',
    "markdown_escapes": lambda t: re.sub(r"([\\`*_{}\[\]()#+\-.!])", r"\\\1", t),
    "homoglyphs": lambda t: t.translate(HOMO),
    "soft_hyphen": lambda t: " ".join(w[: len(w) // 2] + "­" + w[len(w) // 2 :] for w in t.split(" ")),
    "zero_width": lambda t: "​".join(t),
    "hyphen_linebreak": lambda t: " ".join(
        (w[: len(w) // 2] + "-\n" + w[len(w) // 2 :]) if len(w) > 7 else w for w in t.split(" ")
    ),
    "word_per_line": lambda t: "\n".join(t.split()),
    "line_numbers": lambda t: "\n".join(
        f"{n+1:>4} | {line}" for n, line in enumerate(re.findall(r".{1,60}(?:\s|$)", t))
    ),
    "rot13": lambda t: codecs.encode(t, "rot13"),
    "uppercase": lambda t: t.upper(),
    "c_escape_x": lambda t: "".join(f"\\x{ord(c):02x}" if not c.isalnum() else c for c in t),
}
host = "The committee met on Tuesday to review the budget and the new library hours. " * 40
found = defaultdict(int)
for item in items:
    text = text_of(item)
    for name, form in FORMS.items():
        record = host + "\n" + form(text) + "\n" + host
        pairs = record_pairs("r", record, table, locators)
        found[name] += any(p.item == item["id"] for p in pairs)
for name in FORMS:
    print(f"{name:20s} {found[name] / len(items):.3f}")
