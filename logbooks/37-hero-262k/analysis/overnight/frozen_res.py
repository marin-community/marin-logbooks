"""Read-only: frozen-bias arm against its global188 control."""
import json,sys,statistics as st,numpy as np,wandb
S=sys.argv[1]; api=wandb.Api(timeout=120); R="marin-community/marin_moe/"
EXCL=("ptb","wikitext_103","gab","twitterAAE_HELM_fixed")
nm=lambda k:k.split("/")[-1].replace("-llama3","")
def evalm(run):
    w={k:v for k,v in api.run(R+run).summary._json_dict.items() if isinstance(v,(int,float))}; o={}
    for L in (4096,65536,262144):
        p=f"eval_dropless/seq_len_{L}/"; sub={k[len(p):-5]:v for k,v in w.items() if k.startswith(p) and k.endswith("/loss") and k!=p+"loss" and k.count("/")>=3}
        assert len(sub)==23; c={k:v for k,v in sub.items() if nm(k) not in EXCL}; assert len(c)==19
        o[L]={"clean":sum(c.values())/19,"macro":w[p+"macro_loss"],"pal":w[p+"paloma/macro_loss"],"unc":w[p+"uncheatable_eval/macro_loss"],"pm":w[p+"paloma/micro_loss"],"sub":sub}
    return o
F=evalm("hero-mrcr-lce11-frozen-global188-step390351"); C=evalm("hero-mrcr-lce11-merged-global188-step390351"); B=evalm("hero-mrcr-lce11-merged-global188-base")
for L in F:
    print(L,*["%s %.4f ctl %.4f d %+.4f base %.4f"%(k,F[L][k],C[L][k],F[L][k]-C[L][k],B[L][k]) for k in("clean","macro","pal","unc","pm")],sep=" | ")
    dd={k:F[L]["sub"][k]-C[L]["sub"][k] for k in C[L]["sub"]}; cl={k:v for k,v in dd.items() if nm(k) not in EXCL}
    print("  subsets all23 lower %d higher %d mean %+.5f; clean19 lower %d higher %d mean %+.5f; excl %s; biggest clean %s"%(sum(v<0 for v in dd.values()),sum(v>0 for v in dd.values()),st.mean(dd.values()),sum(v<0 for v in cl.values()),sum(v>0 for v in cl.values()),st.mean(cl.values()),{nm(k)[:8]:round(v,4) for k,v in dd.items() if nm(k) in EXCL},[(nm(k)[:12],round(v,4)) for k,v in sorted(cl.items(),key=lambda x:-abs(x[1]))[:3]]))
K=("moe/drop_fraction","train/cross_entropy_loss","train/grad_norm","train/grad_clipped","throughput/iteration_time","memory/peak_gib","moe_bias/applied_norm","moe_bias/solver_norm")
def ser(run):
    r=api.run(R+run); return r,{k:{int(x["_step"]):x[k] for x in r.history(keys=[k],samples=400,pandas=False) if isinstance(x.get(k),(int,float))} for k in K}
rf,f=ser("hero-lce11-frozen-global188"); rc,c=ser("hero-lce11-merged-global188")
print("state",rf.state,"steps",len(f["moe/drop_fraction"]))
d,dc=f["moe/drop_fraction"],c["moe/drop_fraction"]; ks=sorted(d); assert len(ks)==100
for n,q in (("all",ks),("first20",ks[:20]),("mid",ks[20:80]),("last20",ks[80:])):
    print(n,"drop mean %.3e ctl %.3e ratio %.3f max %.3e ctl %.3e"%(st.mean(d[k] for k in q),st.mean(dc[k] for k in q),st.mean(d[k] for k in q)/st.mean(dc[k] for k in q),max(d[k] for k in q),max(dc[k] for k in q)))
x=np.arange(100); y=np.array([d[k] for k in ks]); yc=np.array([dc[k] for k in ks]); print("slope per 100 steps frozen %.2e ctl %.2e ; steps frozen>ctl %d ; per-step ratio max %.2f; 10-step ratio max %.2f"%(np.polyfit(x,y,1)[0]*100,np.polyfit(x,yc,1)[0]*100,(y>yc).sum(),(y/yc).max(),max(y[i:i+10].mean()/yc[i:i+10].mean() for i in range(91))))
print("per10 frozen",[round(y[i:i+10].mean()*1e4,2) for i in range(0,100,10)]); print("per10 ctl   ",[round(yc[i:i+10].mean()*1e4,2) for i in range(0,100,10)])
ce,cc=f["train/cross_entropy_loss"],c["train/cross_entropy_loss"]; de=np.array([ce[k]-cc[k] for k in ks]); print("CE diff mean %.2e first20 %.2e mid %.2e last20 %.2e min %.2e max %.2e; negative %d; per10 e-4 %s"%(de.mean(),de[:20].mean(),de[20:80].mean(),de[80:].mean(),de.min(),de.max(),(de<0).sum(),[round(de[i:i+10].mean()*1e4,2) for i in range(0,100,10)]))
print("CE mean frozen %.4f ctl %.4f"%(st.mean(ce.values()),st.mean(cc.values())))
g,gc=f["train/grad_norm"],c["train/grad_norm"]; print("grad med %.3f ctl %.3f mean %.3f ctl %.3f top %s ctl %s clipped %d ctl %d first20 %d ctl %d"%(st.median(g.values()),st.median(gc.values()),st.mean(g.values()),st.mean(gc.values()),[(a,round(b,2)) for a,b in sorted(g.items(),key=lambda z:-z[1])[:4]],[(a,round(b,2)) for a,b in sorted(gc.items(),key=lambda z:-z[1])[:4]],sum(f["train/grad_clipped"].values()),sum(c["train/grad_clipped"].values()),sum(f["train/grad_clipped"][k] for k in ks[:20]),sum(c["train/grad_clipped"][k] for k in ks[:20])))
print("s/step med %.1f ctl %.1f; peak %.2f ctl %.2f"%(st.median(f["throughput/iteration_time"].values()),st.median(c["throughput/iteration_time"].values()),max(f["memory/peak_gib"].values()),max(c["memory/peak_gib"].values())))
a,s=f["moe_bias/applied_norm"],f["moe_bias/solver_norm"]; print("applied %.5f..%.5f n=%d; solver first %.2f last %.2f mean %.2f min %.2f max %.2f; solver per10 %s"%(min(a.values()),max(a.values()),len(a),s[ks[0]],s[ks[-1]],st.mean(s.values()),min(s.values()),max(s.values()),[round(np.mean([s[k] for k in ks[i:i+10]]),2) for i in range(0,100,10)]))
print("layer keys",[k for k in rf.summary._json_dict if ('layer' in k and 'drop' in k.lower()) or k.startswith('moe/')][:12])
D=S+"/mrcr"; ef={r["source_id"]:r for r in map(json.loads,open(D+"/frozen/mrcr_example_losses.jsonl"))}; ec={r["source_id"]:r for r in map(json.loads,open(D+"/global188/mrcr_example_losses.jsonl"))}
mf=json.load(open(D+"/frozen/mrcr_metrics.json"))["metrics"]; mc=json.load(open(D+"/global188/mrcr_metrics.json"))["metrics"]; PRE="eval/mrcr/two_shot/cap_262144/"
assert set(ef)==set(ec) and json.load(open(D+"/frozen/mrcr_metrics.json"))["checkpoint_step"]==390351
rng=np.random.default_rng(20261106)
for name,sel in (("all",lambda r:True),("far",lambda r:r["distance_band"]=="distance_131073_plus"),("8n",lambda r:r["n_needles"]==8),("8nfar",lambda r:r["n_needles"]==8 and r["distance_band"]=="distance_131073_plus"),("2n",lambda r:r["n_needles"]==2)):
    ids=sorted(i for i in ec if sel(ec[i])); t=np.array([ec[i]["scored_tokens"] for i in ids],float); idx=rng.integers(0,len(ids),(10000,len(ids)))
    A=lambda e,k:np.array([e[i][k] for i in ids],float)
    def est(e,ix=None):
        ff,dd,qq,nn,tt=[A(e,k) for k in("full_context_loss_sum","distractor_only_loss_sum","query_only_loss_sum","needle_only_loss_sum")]+[t]
        if ix is not None: ff,dd,qq,nn,tt=ff[ix],dd[ix],qq[ix],nn[ix],tt[ix]
        return ff.sum(-1)/tt.sum(-1),dd.sum(-1)/tt.sum(-1),(dd-ff).sum(-1)/(qq-nn).sum(-1)
    a0,b0=est(ef),est(ec); a1,b1=est(ef,idx),est(ec,idx)
    print(name,len(ids),*["%s %.4f ctl %.4f d %+.4f [%+.4f,%+.4f]"%(n,a0[i],b0[i],a0[i]-b0[i],*np.percentile(a1[i]-b1[i],[2.5,97.5])) for i,n in enumerate(("full","dist","rec"))],sep=" | ")
pre=PRE
print("rec ci frozen",mf[pre+"micro_needle_recovery_ratio_ci95_low"],mf[pre+"micro_needle_recovery_ratio_ci95_high"],"ctl",mc[pre+"micro_needle_recovery_ratio_ci95_low"],mc[pre+"micro_needle_recovery_ratio_ci95_high"])
