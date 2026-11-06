import json,wandb,re,numpy as np
api=wandb.Api(timeout=120)
r=api.run("marin-community/marin_moe/hero-lce11-merged-keep166")
ks=[k for k in r.summary._json_dict if k.startswith("eval") or "/macro_loss" in k or "eval" in k.split("/")[0]]
print(len(ks),ks[:12],[k for k in ks if k.endswith("macro_loss")][:12])
p=[k for k in ks if k.endswith("/macro_loss") or k.endswith("eval/loss")]
print(p)
EX=("ptb","wikitext_103","gab","twitterAAE_HELM_fixed")
R={"keep166":"hero-lce11-merged-keep166","keep166-raw":"hero-lce11-keep166-raw","log65536sq":"hero-lce11-q64-log65536sq","global188":"hero-lce11-merged-global188","log32768":"hero-lce11-merged-log32768","log65536a20":"hero-lce11-merged-log65536a20","global200":"hero-lce11-merged-global200"}
ROOT=None
sub=[k for k in ks if k.endswith("/loss") and "-llama3" in k]
print("subs",len(sub),sub[:3])
out={}
for a,n in R.items():
    r=api.run("marin-community/marin_moe/"+n); rows={}
    for i in range(0,len(sub),40):
        for x in r.history(keys=sub[i:i+40],samples=20,pandas=False):
            rows.setdefault(int(x["_step"]),{}).update({k:x[k] for k in sub[i:i+40] if isinstance(x.get(k),(int,float))})
    res={}
    for s,v in sorted(rows.items()):
        kept=[w for k,w in v.items() if k.split("/")[-2].replace("-llama3","") not in EX]
        res[s]={"n":len(v),"clean":sum(kept)/len(kept) if kept else None,"macro":sum(v.values())/len(v) if v else None}
    out[a]=res; print(a,{s:(x["n"],round(x["clean"],4)) for s,x in res.items()})
json.dump(out,open("%s/inloop.json"%__import__("sys").argv[1],"w"))
# 11-rack repeat candidates
rs=api.runs("marin-community/marin_moe",filters={"display_name":{"$regex":"^(hero-)?eval-oom-11rack"}})
S={}
for r in rs:
    h={int(x["_step"]):x["train/cross_entropy_loss"] for x in r.history(keys=["train/cross_entropy_loss"],samples=400,pandas=False) if isinstance(x.get("train/cross_entropy_loss"),(int,float))}
    print(r.name,r.state,len(h),(min(h),max(h)) if h else None,{k:r.config.get("model",{}).get(k) for k in("qk_mult","qk_mult_long_scale")},r.tags[:4],(r.commit if hasattr(r,'commit') else None))
    S[r.name]=h
n=list(S)
for i,a in enumerate(n):
    for b in n[i+1:]:
        c=sorted(set(S[a])&set(S[b]))
        if c:
            d=np.array([S[a][k]-S[b][k] for k in c]); print(a,b,"overlap",len(c),c[:2],c[-1],"rms %.2e mean %.2e first5"%(np.sqrt((d**2).mean()),d.mean()),np.round(d[:5],6),"rms by 10s",[float("%.1e"%np.sqrt((d[i:i+10]**2).mean())) for i in range(0,len(d),10)][:6])
