"""Read-only: macro and per-subset held-out loss from W&B eval runs. usage: macro_res.py out.json"""
import json,sys,wandb
api=wandb.Api(timeout=120)
ARMS={"keep166":("hero-mrcr-lce11-merged-keep166-step390351","hero-mrcr-lce11-merged-keep166-base"),
"keep166-raw":("hero-mrcr-lce11-keep166-raw-step390351","hero-mrcr-keep166-step390251"),
"log65536sq":("hero-mrcr-lce11-q64-log65536sq-step390351","hero-mrcr-lce11-merged-log65536sq-base")}
for a in("global188","log32768","log65536a20","global200"): ARMS[a]=(f"hero-mrcr-lce11-merged-{a}-step390351",f"hero-mrcr-lce11-merged-{a}-base")
def get(run):
    r=api.run(f"marin-community/marin_moe/{run}"); assert r.state=="finished",run
    w=r.summary._json_dict; o={"run":run}
    for L in(4096,65536,262144):
        p=f"eval_dropless/seq_len_{L}/"; d={}
        for k in("macro_loss","paloma/macro_loss","uncheatable_eval/macro_loss","paloma/micro_loss","uncheatable_eval/micro_loss","loss"): d[k]=w[p+k]
        d["sub"]={k[len(p):-5]:v for k,v in w.items() if k.startswith(p) and k.endswith("/loss") and k!=p+"loss" and isinstance(v,(int,float))}
        assert len(d["sub"])==23,(run,L,len(d["sub"]))
        o[str(L)]=d
    return o
out={a:{"final":get(e),"base":get(b)} for a,(e,b) in ARMS.items()}
json.dump(out,open(sys.argv[1],"w"),indent=1)
