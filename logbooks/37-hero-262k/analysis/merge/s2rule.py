import json,sys,statistics as st
d=json.load(open(sys.argv[1])); res=d['res']; bs=d['base']; lens=sys.argv[2].split(','); ref=sys.argv[3]
R={}
for k,v in res.items():
    p=k.split('/')
    if len(p)==6 and p[-1]=='loss' and p[2].startswith('seq_len_'): R.setdefault((p[1],p[2][8:]),{})[p[3]+'/'+p[4]]=v
names=sorted({n for n,_ in R}); print(names)
L0=lens[0]; subs=sorted(R[(ref,L0)]); OUT=('ptb','wikitext_103','gab','twitterAAE')
clean=[s for s in subs if not s.split('/')[1].startswith(OUT)]; pal=[s for s in clean if s.startswith('paloma/')]; unc=[s for s in clean if not s.startswith('paloma/')]
print(len(subs),len(clean),len(pal),len(unc))
m=lambda n,L,S:st.mean(R[(n,L)][s] for s in S)
for L in lens:
    bv={s:bs[f'eval_dropless/seq_len_{L}/{s}/loss'] for s in subs}
    print('GUARD',L,'clean',round(m(ref,L,clean)-st.mean(bv[s] for s in clean),6),'off23',round(m(ref,L,subs)-st.mean(bv.values()),6),'raw clean',round(m(ref,L,clean),5),'off',round(m(ref,L,subs),5),'unc',round(m(ref,L,unc),5))
MR='mrcr/two_shot/cap_262144/'
for v in ('micro','macro'):
    key=MR+f'{v}_needle_recovery_ratio'
    print(v,'base',round(bs['eval/'+key],5),{n:round(res.get(f'eval_merge/{n}/{key}',float('nan')),5) for n in names})
for n in names:
    if n==ref: continue
    for L in lens:
        d0={s:R[(n,L)][s]-R[(ref,L)][s] for s in clean}; top=min(d0,key=d0.get)
        print(n,L,'clean',round(st.mean(d0.values()),5),'pal',round(st.mean(d0[s] for s in pal),5),'unc',round(st.mean(d0[s] for s in unc),5),'imp',sum(x<0 for x in d0.values()),'worst',round(max(d0.values()),5),'droptop',round(st.mean(v for s,v in d0.items() if s!=top),5))
