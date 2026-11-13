"""Compare blind labels with the frozen filter's verdicts on the held-out slice.

usage: val_score.py <out dir>
"""

import collections
import json
import sys

out_dir = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "validation"
NAME = {"J": "junk", "D": "data", "K": "keep"}
labels = {}
for line in open(f"{out_dir}/{prefix}_labels.txt"):
    if line.strip() and not line.startswith("#"):
        number, label = line.split()
        labels[int(number)] = NAME[label]
key = [json.loads(line) for line in open(f"{out_dir}/{prefix}_key.jsonl")]
preds = [json.loads(line) for line in open(f"{out_dir}/{prefix}_pred.jsonl")]
population = collections.Counter(p["tier"] for p in preds)
total = sum(population.values())

table = collections.Counter()
drawn = collections.Counter()
misses = []
for k in key:
    label = labels[k["number"]]
    table[k["tier"], label] += 1
    drawn[k["tier"]] += 1
    if (k["tier"] == "keep") != (label == "keep"):
        misses.append((k["number"], k["tier"], k["category"], label, k["junk_share"]))

print(f"held-out population: {total} documents; predicted " + ", ".join(f"{t} {population[t]} ({population[t]/total:.1%})" for t in ("junk", "data", "keep")))
print("\nfilter verdict (rows) x blind label (columns), documents")
print(f"{'':8s} {'junk':>6s} {'data':>6s} {'keep':>6s}   n")
for tier in ("junk", "data", "keep"):
    print(f"{tier:8s} " + " ".join(f"{table[tier, label]:6d}" for label in ("junk", "data", "keep")) + f"   {drawn[tier]}")

# Each stratum's sample rate differs, so weight back to the population.
weight = {tier: population[tier] / drawn[tier] for tier in drawn}
flagged_true = sum(weight[t] * (table[t, "junk"] + table[t, "data"]) for t in ("junk", "data"))
flagged_all = sum(weight[t] * drawn[t] for t in ("junk", "data"))
missed = weight["keep"] * (table["keep", "junk"] + table["keep", "data"])
print(f"\nflagged documents that are junk or data by label: {flagged_true/flagged_all:.1%}")
print(f"  machine-junk verdicts labelled junk or data: {(table['junk','junk']+table['junk','data'])/drawn['junk']:.1%}; labelled junk: {table['junk','junk']/drawn['junk']:.1%}")
print(f"  data verdicts labelled junk or data: {(table['data','junk']+table['data','data'])/drawn['data']:.1%}")
print(f"unflagged documents labelled junk or data: {(table['keep','junk']+table['keep','data'])/drawn['keep']:.1%}")
print(f"estimated share of all junk-or-data documents the filter flags: {flagged_true/(flagged_true+missed):.1%}")
junk_caught = sum(weight[t] * table[t, "junk"] for t in ("junk", "data"))
junk_missed = weight["keep"] * table["keep", "junk"]
print(f"estimated share of machine-junk documents flagged (either tier): {junk_caught/(junk_caught+junk_missed):.1%}")
print("\ndisagreements on flag versus keep (sheet number, verdict, category, label, junk share):")
for m in sorted(misses):
    print("  ", m)
