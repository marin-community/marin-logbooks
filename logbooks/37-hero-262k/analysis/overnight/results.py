import itertools, json, os, sys
import numpy as np, wandb
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
configure_coreweave_s3()
ARMS = "keep166 global188 global200 log32768 log65536a20 log65536sq".split()
V = "2026.11.05"; P = "marin-us-east-02a/marin/eval/mrcr-hero/"; PRE = "eval/mrcr/two_shot/cap_262144/"
CONDS = ("full_context", "distractor_only", "needle_only", "query_only")
EXCL = ("ptb", "wikitext_103", "gab", "twitterAAE_HELM_fixed")
K = 1024; PB = [(0,16),(16,32),(32,48),(48,64),(64,80),(80,96),(96,128),(128,192),(192,256)]; BOOT = 10000
fs = filesystem_for("s3://" + P); fs = fs[0] if isinstance(fs, tuple) else fs; fs.invalidate_cache()
man = {}
for l in fs.cat(P + "data/two_shot-cap262144/2026.11.04/manifest.jsonl").decode().splitlines():
    x = json.loads(l); man.setdefault(x["source_id"], {})[x["condition"]] = x
api = wandb.Api(timeout=120)
M, R, W = {}, {}, {}
for a in ARMS:
    for s in (390251, 390351):
        run = f"hero-mrcr-{a}-step{s}"; base = f"{P}{run}/{V}/"
        M[a, s] = json.loads(fs.cat(base + "mrcr_metrics.json"))
        R[a, s] = {x["source_id"]: x for x in map(json.loads, fs.cat(base + "mrcr_example_losses.jsonl").decode().splitlines())}
        r = api.run(f"marin-community/marin_moe/{run}")
        assert r.state == "finished", run
        W[a, s] = {k: v for k, v in r.summary._json_dict.items() if isinstance(v, (int, float))}
        m = M[a, s]; assert m["checkpoint_step"] == s and m.get("max_batches") is None, run
ids = sorted(R["keep166", 390251]); assert len(ids) == 299
for k, v in R.items(): assert sorted(v) == ids, k
sc = np.array([R["keep166", 390251][i]["scored_tokens"] for i in ids], float)
dist = np.array([R["keep166", 390251][i]["evidence_distance_tokens"] for i in ids], float)
fl = np.array([man[i]["full_context"]["length_tokens"] for i in ids], float)
pos = fl - sc - 2 - dist
bins = [np.where((pos >= a * K) & (pos < b * K))[0] for a, b in PB]; assert sum(map(len, bins)) == 299
C = {k: {c: np.array([v[i][c + "_loss_sum"] for i in ids], float) for c in CONDS} for k, v in R.items()}
rng = np.random.default_rng(20261106)
BI = rng.integers(0, 299, (BOOT, 299))
def ci(v): return [float(x) for x in np.percentile(v, [2.5, 97.5])]
def rec(c, ix=None, bi=None):
    f = (lambda x: x[ix] if ix is not None else x)
    if bi is None:
        return float((f(c["distractor_only"]).sum() - f(c["full_context"]).sum()) / (f(c["query_only"]).sum() - f(c["needle_only"]).sum()))
    s = lambda x: f(x)[bi].sum(1)
    return (s(c["distractor_only"]) - s(c["full_context"])) / (s(c["query_only"]) - s(c["needle_only"]))
def paired(d):
    return {"delta": float(d.sum() / sc.sum()), "ci95": ci(d[BI].sum(1) / sc[BI].sum(1))}
HO = ("macro_loss", "paloma/macro_loss", "uncheatable_eval/macro_loss", "paloma/micro_loss")
def clean(w, L):
    pre = f"eval_dropless/seq_len_{L}/"
    subs = {k[len(pre):-5]: v for k, v in w.items() if k.startswith(pre) and k.endswith("/loss") and k != pre + "loss" and k.count("/") >= 3 and "macro" not in k and "micro" not in k and ("paloma/" in k or "uncheatable_eval/" in k)}
    kept = [v for k, v in subs.items() if k.split("/")[-1].replace("-llama3", "") not in EXCL]
    assert len(subs) == 23 and len(kept) == 19, (L, len(subs), len(kept))
    return sum(kept) / 19
out = {"arms": {}, "note": "delta is step390351 minus step390251; numbers only"}
for a in ARMS:
    o = {}
    for s in (390251, 390351):
        c = C[a, s]; m = M[a, s]["metrics"]
        e = {f"{c_}_nll": float(c[c_].sum() / sc.sum()) for c_ in CONDS}
        e.update({f"{c_}_ci95": [m[f"{PRE}{c_}/micro_loss_ci95_low"], m[f"{PRE}{c_}/micro_loss_ci95_high"]] for c_ in CONDS})
        e["recovery"] = m[PRE + "micro_needle_recovery_ratio"]; e["recovery_ci95"] = [m[PRE + "micro_needle_recovery_ratio_ci95_low"], m[PRE + "micro_needle_recovery_ratio_ci95_high"]]
        assert abs(rec(c) - e["recovery"]) < 1e-5, (a, s)
        e["recovery_by_position"] = {}
        for (x, y), ix in zip(PB, bins):
            bi = rng.integers(0, len(ix), (BOOT, len(ix)))
            e["recovery_by_position"][f"{x}_{y}"] = {"value": rec(c, ix), "ci95": ci(rec(c, ix, bi)), "n": int(len(ix))}
        for L in (4096, 65536, 262144):
            w = W[a, s]
            for h in HO: e[f"{h}@{L}"] = w[f"eval_dropless/seq_len_{L}/{h}"]
            e[f"clean19@{L}"] = clean(w, L)
        o[s] = e
    o["step_delta_paired"] = {c_: paired(C[a, 390351][c_] - C[a, 390251][c_]) for c_ in CONDS}
    b = o[390251]; n = o[390351]
    o["delta"] = {k: n[k] - b[k] for k in b if isinstance(b[k], float)}
    o["checkpoint"] = M[a, 390351]["checkpoint_path"]
    o["code"] = {"qk_mult": M[a, 390351]["qk_mult"], "long": M[a, 390351]["qk_mult_long_scale"], "floor": M[a, 390351]["log_scaling_n_floor"], "alpha": M[a, 390351].get("log_scaling_alpha"), "power": M[a, 390351].get("log_scaling_power")}
    out["arms"][a] = o
out["paired_arm_deltas_390351"] = {f"{a} - {b}": {c_: paired(C[a, 390351][c_] - C[b, 390351][c_]) for c_ in ("full_context", "distractor_only")} for a, b in itertools.combinations(ARMS, 2)}
out["paired_arm_deltas_390251"] = {f"{a} - {b}": {c_: paired(C[a, 390251][c_] - C[b, 390251][c_]) for c_ in ("full_context", "distractor_only")} for a, b in itertools.combinations(ARMS, 2)}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1])
