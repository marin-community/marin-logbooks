"""Read-only: recovery and full NLL by MRCR group, with paired bootstrap against Keep 1.66 merged. usage: hard_res.py mrcr_dir out.json"""
import json,sys,numpy as np
D,OUT=sys.argv[1],sys.argv[2]
ARMS=("keep166","keep166-raw","log65536sq","global188","log32768","log65536a20","global200")
CAND=("keep166","global188","global200","log32768")
BANDS=("distance_0_32768","distance_32769_65536","distance_65537_131072","distance_131073_plus")
PRE="eval/mrcr/two_shot/cap_262144/"
ex={a:{r["source_id"]:r for r in map(json.loads,open(f"{D}/{a}/mrcr_example_losses.jsonl"))} for a in ARMS}
mx={a:json.load(open(f"{D}/{a}/mrcr_metrics.json"))["metrics"] for a in ARMS}
ids=sorted(ex["keep166"]); assert all(set(ex[a])==set(ids) for a in ARMS)
ref=ex["keep166"]; assert all(ex[a][i]["n_needles"]==ref[i]["n_needles"] and ex[a][i]["distance_band"]==ref[i]["distance_band"] and ex[a][i]["scored_tokens"]==ref[i]["scored_tokens"] for a in ARMS for i in ids)
groups={"all":ids}
for b in BANDS: groups[b]=[i for i in ids if ref[i]["distance_band"]==b]
for n in (2,4,8):
    groups[f"{n}needle"]=[i for i in ids if ref[i]["n_needles"]==n]
    for b in BANDS: groups[f"{n}needle/{b}"]=[i for i in ids if ref[i]["n_needles"]==n and ref[i]["distance_band"]==b]
def arr(a,g,k): return np.array([ex[a][i][k] for i in g],float)
def stats(a,g):
    return arr(a,g,"full_context_loss_sum"),arr(a,g,"distractor_only_loss_sum"),arr(a,g,"query_only_loss_sum"),arr(a,g,"needle_only_loss_sum")
def est(s,t,idx=None):
    f,d,q,n=s; ax=-1
    if idx is not None: f,d,q,n,t=f[idx],d[idx],q[idx],n[idx],t[idx]
    nll=f.sum(ax)/t.sum(ax); orc=(q-n).sum(ax); real=(d-f).sum(ax)
    with np.errstate(divide="ignore",invalid="ignore"): rec=real/orc
    return nll,rec,d.sum(ax)/t.sum(ax),orc/t.sum(ax)
rng=np.random.default_rng(20261106); out={"groups":{}}
for name,g in groups.items():
    t=arr("keep166",g,"scored_tokens"); idx=rng.integers(0,len(g),(10000,len(g)))
    pre=PRE+("" if name=="all" else name+"/")
    cell={"n":len(g),"tokens":float(t.sum()),"arms":{},"delta":{},"pairs":{}}
    S={a:stats(a,g) for a in ARMS}; B={a:est(S[a],t,idx) for a in ARMS}
    for a in ARMS:
        nll,rec,dn,orc=est(S[a],t); m=mx[a]
        cell["arms"][a]={"full":float(nll),"full_ci":[m[pre+"full_context/micro_loss_ci95_low"],m[pre+"full_context/micro_loss_ci95_high"]],"dist":float(dn),"rec":float(rec),
          "rec_ci":[m.get(pre+"micro_needle_recovery_ratio_ci95_low"),m.get(pre+"micro_needle_recovery_ratio_ci95_high")],"identifiable":m[pre+"micro_recovery_identifiable"],"oracle":float(orc)}
        assert abs(nll-m[pre+"full_context/micro_loss"])<1e-9
    def dl(a,c):
        f=B[a][0]-B[c][0]; r=B[a][1]-B[c][1]; fe=est(S[a],t)[0]-est(S[c],t)[0]; re=est(S[a],t)[1]-est(S[c],t)[1]
        return {"full":[float(fe),*map(float,np.percentile(f,[2.5,97.5]))],"rec":[float(re),*map(float,np.nanpercentile(r,[2.5,97.5]))]}
    for a in ARMS:
        if a!="keep166": cell["delta"][a]=dl(a,"keep166")
    for i,a in enumerate(CAND):
        for c in CAND[i+1:]: cell["pairs"][f"{c}|{a}"]=dl(c,a)
    out["groups"][name]=cell
json.dump(out,open(OUT,"w"),indent=1)
fm=lambda v:"%+.4f[%+.4f,%+.4f]%s"%(*v,"*" if v[1]*v[2]>0 else "")
for name in ("all","distance_131073_plus","8needle","8needle/distance_131073_plus","distance_65537_131072","4needle/distance_131073_plus","2needle/distance_131073_plus"):
    c=out["groups"][name];print(name,"n",c["n"],"tokens",c["tokens"])
    for a in ARMS:
        x=c["arms"][a];print("  ",a,"full %.4f[%.4f,%.4f] rec %.3f %s id %s"%(x["full"],*x["full_ci"],x["rec"],x["rec_ci"],x["identifiable"]),"" if a=="keep166" else "dF "+fm(c["delta"][a]["full"])+" dR "+fm(c["delta"][a]["rec"]))
    print("  cand pairs",{k:(fm(v["full"]),fm(v["rec"])) for k,v in c["pairs"].items() if v["full"][1]*v["full"][2]>0 or v["rec"][1]*v["rec"][2]>0})
tot=sum(1 for n,c in out["groups"].items() if n!="all" for a,d in c["delta"].items() if a in("global188","global200","log32768") for m in("full","rec")); sig=[(n,a,m,round(d[m][0],4)) for n,c in out["groups"].items() if n!="all" for a,d in c["delta"].items() if a in("global188","global200","log32768") for m in("full","rec") if d[m][1]*d[m][2]>0]
print("cand vs keep all groups: tests",tot,"excl0",len(sig),sig)
print("min n",sorted((c["n"],n) for n,c in out["groups"].items())[:6]); print("nonident",[(n,a) for n,c in out["groups"].items() for a,x in c["arms"].items() if not x["identifiable"]])
