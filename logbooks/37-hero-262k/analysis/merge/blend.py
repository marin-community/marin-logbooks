import wandb,json,sys,statistics as st
S=sys.argv[1]
api=wandb.Api(timeout=120)
r=api.run("marin-community/marin_moe/hero-merge-blend-step390351"); print(r.state)
res={}
for row in r.scan_history(page_size=1000):
    for k,v in row.items():
        if k.startswith('eval_merge/') and not k.startswith('eval_merge/distance') and isinstance(v,(int,float)): res.setdefault(k,v)
base={}
for n,rid in (('w000','hero-mrcr-keep166-step390251'),('w100','hero-mrcr-keep166-step390351')):
    base[n]={k:v for k,v in api.run("marin-community/marin_moe/"+rid).summary._json_dict.items() if isinstance(v,(int,float))}
json.dump(dict(res=res,base=base),open(S+'/blend_raw.json','w'))
names=['w000','w025','w050','w075','w090','w100']; print(sorted({k.split('/')[1] for k in res}))
Ls=['4096','262144']
R={}
for k,v in res.items():
    p=k.split('/')
    if len(p)==6 and p[-1]=='loss' and p[2].startswith('seq_len_'): R.setdefault((p[1],p[2][8:]),{})[p[3]+'/'+p[4]]=v
subs=sorted(R[('w000','4096')]); OUT=('ptb','wikitext_103','gab','twitterAAE')
clean=[s for s in subs if not s.split('/')[1].startswith(OUT)];pal=[s for s in clean if s.startswith('paloma/')];unc=[s for s in clean if not s.startswith('paloma/')];noptb=[s for s in subs if 'ptb' not in s]
print(len(subs),len(clean),len(pal),len(unc),len(noptb))
m=lambda n,L,S2:st.mean(R[(n,L)][s] for s in S2)
print('GUARDS')
for n in ('w000','w100'):
    bs=base[n]
    for L in Ls:
        b={s:bs[f'eval_dropless/seq_len_{L}/{s}/loss'] for s in subs}
        print(n,L,'clean',round(m(n,L,clean)-st.mean(b[s] for s in clean),6),'no-ptb22',round(m(n,L,noptb)-st.mean(b[s] for s in noptb),6),'full23',round(m(n,L,subs)-st.mean(b.values()),6),'ptb',round(R[(n,L)]['paloma/ptb-llama3']-b['paloma/ptb-llama3'],5))
    for v in('micro','macro'):
        key=f'mrcr/two_shot/cap_262144/{v}_needle_recovery_ratio'
        print(' ',n,v,round(res[f'eval_merge/{n}/{key}'],5),'base',round(bs['eval/'+key],5))
print('RECOVERY')
for v in('micro','macro'):
    key=f'mrcr/two_shot/cap_262144/{v}_needle_recovery_ratio'
    Rv={n:res[f'eval_merge/{n}/{key}'] for n in names}
    print(v,{n:round(x,5) for n,x in Rv.items()})
    print('  f',{n:round((Rv[n]-Rv['w000'])/(Rv['w100']-Rv['w000']),4) for n in names})
print('other MRCR keys',[k for k in res if k.startswith('eval_merge/w000/mrcr') and k.count('/')<=5 and 'recovery_ratio' in k and 'ci95' not in k][:12])
for L in Ls:
  print('HELD-OUT',L)
  for n in names:
    d={s:R[(n,L)][s]-R[('w000',L)][s] for s in clean}
    print(n,'abs',round(m(n,L,clean),5),'D',round(st.mean(d.values()),5),'pal',round(st.mean(d[s] for s in pal),5),'unc',round(st.mean(d[s] for s in unc),5),'imp',sum(x<0 for x in d.values()),'worst',round(max(d.values()),5))
  print(' ptb',{n:round(R[(n,L)]['paloma/ptb-llama3'],5) for n in names}, 'full23',{n:round(m(n,L,subs),5) for n in names})
