"""Read-only: collect finished arms' results into arms_res.json. usage: arms_res.py out.json arm..."""
import asyncio, json, os, statistics as st, sys
import numpy as np, tensorstore as ts, wandb
from levanter.tensorstore_serialization import _create_ocdbt_spec
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
configure_coreweave_s3()
OUT, ARMS = sys.argv[1], sys.argv[2:]
V="2026.11.05"; P="marin-us-east-02a/marin/eval/mrcr-hero/"; PRE="eval/mrcr/two_shot/cap_262144/"
EXCL=("ptb","wikitext_103","gab","twitterAAE_HELM_fixed")
RAW="s3://marin-us-east-02a/marin/grug/hero-ctx65k-step372k/2026.08.19.2/checkpoints/step-390251"; M="s3://marin-us-east-02a/marin/grug/hero-ctx65k-merge-hb-avg11/checkpoints/step-390251"
LEAF="params/stacked_blocks/stacked/mlp/router_bias"
fs=filesystem_for("s3://"+P); fs=fs[0] if isinstance(fs,tuple) else fs; fs.invalidate_cache()
api=wandb.Api(timeout=120)
def clean(w,L):
    pre=f"eval_dropless/seq_len_{L}/"
    subs={k[len(pre):-5]:v for k,v in w.items() if k.startswith(pre) and k.endswith("/loss") and k!=pre+"loss" and k.count("/")>=3}
    kept=[v for k,v in subs.items() if k.split("/")[-1].replace("-llama3","") not in EXCL]
    assert len(subs)==23 and len(kept)==19,(L,len(subs),len(kept))
    return sum(kept)/19
def ev(run):
    b=f"{P}{run}/{V}/"; m=json.loads(fs.cat(b+"mrcr_metrics.json")); x=m["metrics"]
    r=api.run(f"marin-community/marin_moe/{run}"); assert r.state=="finished",run
    w={k:v for k,v in r.summary._json_dict.items() if isinstance(v,(int,float))}
    e={"run":run,"step":m["checkpoint_step"],"ckpt":m.get("checkpoint") or m.get("checkpoint_path"),"maxb":m.get("max_batches")}
    for c in ("full_context","distractor_only","needle_only","query_only"): e[c]=x[PRE+c+"/micro_loss"]
    e["rec"]=x[PRE+"micro_needle_recovery_ratio"]; e["rec_ci"]=[x[PRE+"micro_needle_recovery_ratio_ci95_low"],x[PRE+"micro_needle_recovery_ratio_ci95_high"]]
    for L in (4096,65536,262144): e[f"c{L}"]=clean(w,L)
    ex={q["source_id"]:q for q in map(json.loads,fs.cat(b+"mrcr_example_losses.jsonl").decode().splitlines())}
    assert len(ex)==299
    return e,ex
def series(r,k):
    return {int(x["_step"]):x[k] for x in r.history(keys=[k],samples=2000,pandas=False) if isinstance(x.get(k),(int,float))}
async def rd(root):
    return np.asarray(await (await ts.open(_create_ocdbt_spec(root,LEAF),open=True,read=True)).read()).astype(np.float64)
out={"arms":{}}; EX={}
for a in ARMS:
    raw=a=="keep166-raw"
    run=("hero-lce11-keep166-raw" if raw else f"hero-lce11-merged-{a}"); er=("hero-mrcr-lce11-keep166-raw-step390351" if raw else f"hero-mrcr-lce11-merged-{a}-step390351")
    br=("hero-mrcr-keep166-step390251" if raw else f"hero-mrcr-lce11-merged-{a}-base")
    pfx=("s3://marin-us-east-08a/tmp/ttl=1d/hero-lce11-keep166-raw" if raw else f"s3://marin-us-east-08a/tmp/ttl=1d/hero-lce11-merged/{a}")
    if a=="log65536sq":  # runs at loader queue 64 under its own tag, against the merged baseline
        run="hero-lce11-q64-log65536sq"; er="hero-mrcr-lce11-q64-log65536sq-step390351"; pfx="s3://marin-us-east-08a/tmp/ttl=1d/hero-lce11-q64/log65536sq"
    e,x=ev(er); b,bx=ev(br); assert e["step"]==390351 and b["step"]==390251 and e["maxb"] is None,(a,e,b)
    r=api.run(f"marin-community/marin_moe/{run}"); m=r.config.get("model",{})
    S={k:series(r,k) for k in ("train/cross_entropy_loss","train/grad_norm","moe/drop_fraction","throughput/iteration_time","memory/peak_gib")}
    steps=sorted(S["moe/drop_fraction"])
    async def two(): return await asyncio.gather(rd(RAW if raw else M),rd(pfx+"/checkpoints/step-390351"))
    b0,b1=asyncio.run(two())
    out["arms"][a]={"eval":e,"base":b,"train_run":run,"state":r.state,"cfg":{k:m.get(k) for k in("qk_mult","qk_mult_long_scale","log_scaling_n_floor","log_scaling_alpha","log_scaling_power","max_seq_len")},
      "tags":[g for g in r.tags if "skew" in g],"steps":[min(steps),max(steps),len(steps)],"drop_mean":st.mean(S["moe/drop_fraction"].values()),"drop_max":max(S["moe/drop_fraction"].values()),"s_step":st.median(S["throughput/iteration_time"].values()),"peak_gib":max(S["memory/peak_gib"].values()) if S["memory/peak_gib"] else None,
      "series":{k:{str(s):v for s,v in d.items()} for k,d in S.items() if k!="memory/peak_gib"},"bias_base":float(np.linalg.norm(b0)),"bias_final":float(np.linalg.norm(b1)),"bias_dist":float(np.linalg.norm(b1-b0)),"bias_layer_med":float(np.median(np.linalg.norm(b1,axis=1)))}
    EX[a]=x
ids=sorted(next(iter(EX.values()))); sc=np.array([next(iter(EX.values()))[i]["scored_tokens"] for i in ids],float)
BI=np.random.default_rng(20261106).integers(0,299,(10000,299)); pairs={}
for i,a in enumerate(ARMS):
    for c in ARMS[i+1:]:
        for k in ("full_context","distractor_only"):
            d=np.array([EX[c][j][k+"_loss_sum"]-EX[a][j][k+"_loss_sum"] for j in ids],float)
            pairs[f"{c}-{a}|{k}"]=[float(d.sum()/sc.sum()),*map(float,np.percentile(d[BI].sum(1)/sc[BI].sum(1),[2.5,97.5]))]
out["pairs"]=pairs; json.dump(out,open(OUT,"w"),indent=1)
for a,o in out["arms"].items():
    e,b=o["eval"],o["base"]
    print(a,o["state"],o["cfg"],o["tags"],o["steps"])
    print("  clean base/final 4K %.4f %.4f (%+.4f) 65K %.4f %.4f (%+.4f) 262K %.4f %.4f (%+.4f)"%(b["c4096"],e["c4096"],e["c4096"]-b["c4096"],b["c65536"],e["c65536"],e["c65536"]-b["c65536"],b["c262144"],e["c262144"],e["c262144"]-b["c262144"]))
    print("  rec %.3f -> %.3f [%.3f,%.3f] full %.4f -> %.4f dist %.4f -> %.4f"%(b["rec"],e["rec"],*e["rec_ci"],b["full_context"],e["full_context"],b["distractor_only"],e["distractor_only"]))
    print("  drop mean %.2e max %.2e s/step %.1f peak %s bias %.2f -> %.2f dist %.2f"%(o["drop_mean"],o["drop_max"],o["s_step"],o["peak_gib"],o["bias_base"],o["bias_final"],o["bias_dist"]))
    ce=o["series"]["train/cross_entropy_loss"]; gn=o["series"]["train/grad_norm"]; ks=sorted(ce,key=int)
    print("  CE first3",[round(ce[k],3) for k in ks[:3]],"first20 mean %.4f all mean %.4f"%(st.mean(ce[k] for k in ks[:20]),st.mean(ce.values())),"gn first20 max %.3f all mean %.3f max %.3f"%(max(gn[k] for k in ks[:20] if k in gn),st.mean(gn.values()),max(gn.values())))
