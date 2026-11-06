"""Read-only: per-step train grad norm and clip flags from W&B. usage: grad_res.py out.json"""
import json,sys,wandb
R={"keep166":"hero-lce11-merged-keep166","keep166-raw":"hero-lce11-keep166-raw","log65536sq":"hero-lce11-q64-log65536sq","global188":"hero-lce11-merged-global188","log32768":"hero-lce11-merged-log32768","log65536a20":"hero-lce11-merged-log65536a20","global200":"hero-lce11-merged-global200"}
api=wandb.Api(timeout=90); out={}
for a,n in R.items():
    r=api.run("marin-community/marin_moe/"+n); o={}
    for k in("train/grad_norm","train/grad_clipped"):
        o[k]={str(int(x["_step"])):x[k] for x in r.history(keys=[k],samples=400,pandas=False) if isinstance(x.get(k),(int,float))}
    out[a]=o
json.dump(out,open(sys.argv[1],"w"))
for a,o in out.items():
    g,c=o["train/grad_norm"],o["train/grad_clipped"]; ks=sorted(g,key=int)
    print(a,len(ks),"clipped first20",int(sum(c[k] for k in ks[:20])),"all",int(sum(c.values())),"min pre-clip when clipped %.3f max unclipped %.3f"%(min(g[k] for k in ks if c[k]),max(g[k] for k in ks if not c[k])),"max20 %.2f"%max(g[k] for k in ks[:20]),"max after20 %.2f"%max(g[k] for k in ks[20:]))
