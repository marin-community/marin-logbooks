import sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1])
for s in ["110 miles", "Day 1", "0.2 kg × 5", "Step‑by‑step", "low‑data", " ", "‑", " 1", "-"]:
    ids = tk.encode(s, add_special_tokens=False).ids
    print(repr(s), ids, [tk.id_to_token(i) for i in ids])
vocab = tk.get_vocab()
# byte-level: U+202F = e2 80 af -> 'âĢ¯' ; U+2011 = e2 80 91 -> 'âĢĳ'
hits = [t for t in vocab if "âĢ¯" in t or "âĢĳ" in t]
print("vocab tokens containing U+202F/U+2011:", len(hits), hits[:30])
