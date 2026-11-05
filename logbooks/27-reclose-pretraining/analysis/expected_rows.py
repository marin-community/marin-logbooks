import json, sys
from experiments.june_tpu_67b_a2b.moe.launch_datakit_moe_mix import _phase_weights, _TAIL_BUCKETS
art = json.load(open(sys.argv[1])); swe = json.load(open(sys.argv[2]))
cell_tokens = {f"c{b['cluster_id']:02d}q{b['quality_bucket']}": b["total_tokens"] for b in art["buckets"]}
cell_docs = {f"c{b['cluster_id']:02d}q{b['quality_bucket']}": b["total_elements"] for b in art["buckets"]}
w = _phase_weights(1)
tail_tot = sum(cell_tokens.get(c, 0) for c in _TAIL_BUCKETS)
weights = {}
for name, val in w.items():
    if name == "tail":
        for c in _TAIL_BUCKETS:
            weights[c] = weights.get(c, 0) + val * cell_tokens.get(c, 0) / tail_tot
    else:
        weights[name] = weights.get(name, 0) + val
assert abs(sum(weights.values()) - 1) < 1e-6
SEQ, BS, STEPS = 65536, 1024, 100
tok_per_step = SEQ * BS
rows = []
total_share = 0.0
for c, wc in sorted(weights.items(), key=lambda kv: -kv[1]):
    st = swe.get(c, {}).get("tokens", 0)
    share = st / cell_tokens[c] if cell_tokens.get(c) else 0.0
    total_share += wc * share
    if st:
        rows.append((c, wc, cell_tokens[c], st, share, wc * share))
print(f"{'cell':6} {'w1':>8} {'cell_tokens':>14} {'swe_tokens':>13} {'share':>7} {'w*share':>9}")
for c, wc, ct, st, share, contrib in sorted(rows, key=lambda r: -r[5])[:12]:
    print(f"{c:6} {wc:8.4f} {ct:14,d} {st:13,d} {share:7.3%} {contrib:9.5f}")
swe_tok_step = total_share * tok_per_step
print("swe-zero share of training tokens:", f"{total_share:.4%}")
print("per step: tokens", f"{swe_tok_step:,.0f}", "docs", f"{swe_tok_step/8839:,.0f}", "seq-equivalents", f"{swe_tok_step/SEQ:,.1f}")
print(f"last {STEPS} steps: tokens {STEPS*swe_tok_step:,.0f}  docs {STEPS*swe_tok_step/8839:,.0f}  seq-equivalents {STEPS*swe_tok_step/SEQ:,.0f} of {STEPS*BS:,d} sequences")
print("whole 15,000-step cooldown: docs", f"{15000*swe_tok_step/8839:,.0f}", "tokens", f"{15000*swe_tok_step/1e9:.2f}B")
# epochs of the c01 cells over the cooldown
for c in ("c01q3","c01q4","c01q2","c01q1"):
    print(c, "cooldown reads", f"{15000*tok_per_step*weights[c]/1e9:.1f}B tokens of a", f"{cell_tokens[c]/1e9:.1f}B cell ->", f"{15000*tok_per_step*weights[c]/cell_tokens[c]:.2f} epochs")
