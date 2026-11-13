"""Token and training-position shares by verdict tier on the held-out slice. usage: tokshare.py <validation_pred.jsonl>"""

import collections
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
CAP = 262_144
tokens = collections.Counter()
capped = collections.Counter()
positions = collections.Counter()
for r in rows:
    tokens[r["tier"]] += r["tokens"]
    capped[r["tier"]] += min(r["tokens"], CAP)
    positions[r["tier"]] += r["position_weight"]
for tier in ("junk", "data", "keep"):
    print(
        f"{tier:5s} tokens {tokens[tier]/sum(tokens.values()):6.1%}  capped at 262k {capped[tier]/sum(capped.values()):6.1%}  "
        f"training positions {positions[tier]/sum(positions.values()):6.1%}"
    )
