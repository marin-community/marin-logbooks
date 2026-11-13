"""Print per-chunk features for chosen documents. usage: feat.py <junk_filter.py> <replay.jsonl> <i,i,...> [max_chunks]"""

import dataclasses
import importlib.util
import json
import re
import sys

spec = importlib.util.spec_from_file_location("junk_filter", sys.argv[1])
jf = importlib.util.module_from_spec(spec)
sys.modules["junk_filter"] = jf
spec.loader.exec_module(jf)
wanted = {int(v) for v in sys.argv[3].split(",")}
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 6
MARKER = re.compile(r"\n\n\[… [\d,]+ tokens omitted …\]\n\n")
SHORT = {
    "binary_share": "bin", "encoded_share": "enc", "compression_ratio": "zr", "lines": "ln", "mean_line_chars": "mlc",
    "repeated_line_share": "rep", "templated_line_share": "tpl", "lock_markers": "lock", "syntax_share": "syn",
    "escaped_newlines_per_char": "esc", "digit_share": "dig", "letter_share": "let", "separator_share": "sep",
    "json_keys_per_char": "json", "markup_share": "mk", "delimited_line_share": "dl", "key_value_line_share": "kv",
    "log_line_share": "log", "sql_line_share": "sql", "item_line_share": "item",
}
for i, line in enumerate(open(sys.argv[2])):
    if i not in wanted:
        continue
    row = json.loads(line)
    print(f"\n#{i} {row['source']} tokens={row['tokens']:,}")
    for n, chunk in enumerate(jf.chunks(MARKER.split(row["text"]))):
        if n >= limit:
            break
        f = jf.chunk_features(chunk)
        values = dataclasses.asdict(f)
        shown = " ".join(
            f"{SHORT.get(k, k)}={v:.0f}" if k in ("lines", "mean_line_chars", "lock_markers") else f"{SHORT.get(k, k)}={v:.2f}"
            for k, v in values.items()
            if v
        )
        print(f"  [{jf.features_junk_category(f)}] {shown}")
