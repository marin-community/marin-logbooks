import json,sys,statistics as st
raw=json.load(open(sys.argv[1]))
Ls=['seq_len_4096','seq_len_65536']
R={}
for k,v in raw.items():
    p=k.split('/')
    if len(p)==6 and p[-1]=='loss' and p[2] in Ls: R.setdefault((p[1],p[2]),{})[p[3]+'/'+p[4]]=v
names=sorted({n for n,_ in R})
print('have',names)
subs=sorted(R[('step-390000',Ls[0])]); OUT=('ptb','wikitext_103','gab','twitterAAE')
clean=[s for s in subs if not s.split('/')[1].startswith(OUT)]
pal=[s for s in clean if s.startswith('paloma/')]; unc=[s for s in clean if not s.startswith('paloma/')]
print(len(subs),len(clean),len(pal),len(unc))
def m(n,L,S): return st.mean(R[(n,L)][s] for s in S)
chain=[390000,384000,378000,372000,366000,360000]
for L in Ls:
    print('==',L,'macro raw clean/pal/unc; official')
    for s in chain:
        n=f'step-{s}'
        if (n,L) in R: print(n,round(m(n,L,clean),5),round(m(n,L,pal),5),round(m(n,L,unc),5),round(m(n,L,subs),5), 'vsT',round(m(n,L,clean)-m('step-390000',L,clean),5))
T='step-390000'
D={}
for k in range(2,7):
    W=f'X{k}'
    if (W,Ls[0]) not in R: continue
    print('==',W)
    for L in Ls:
        d={s:R[(W,L)][s]-R[(T,L)][s] for s in clean}
        rs=[f'step-{x}' for x in chain[:k]]
        lag=st.mean(m(r,L,clean) for r in rs)-m(T,L,clean)
        top=min(d,key=d.get)
        row=dict(clean=st.mean(d.values()),pal=st.mean(d[s] for s in pal),unc=st.mean(d[s] for s in unc),imp=sum(x<0 for x in d.values()),worst=max(d.values()),lag=lag,G=st.mean(d.values())-lag,droptop=st.mean(v for s,v in d.items() if s!=top))
        D[(W,L)]=row
        print(L,{a:(round(b,5) if isinstance(b,float) else b) for a,b in row.items()})
for k in (4,5,6):
    W=f'X{k}'
    if (W,Ls[0]) not in R: continue
    out={}
    for g in ('clean','pal','unc'):
        C={L:D[(W,L)][g]-D[('X3',L)][g] for L in Ls}
        out[g]=dict(C4=round(C[Ls[0]],5),C65=round(C[Ls[1]],5),I=round(C[Ls[1]]-C[Ls[0]],5))
    print(W,out)
print('SCREEN')
have=[f'step-{s}' for s in chain[::-1] if (f'step-{s}',Ls[0]) in R]
for L in Ls:
    for s in subs:
        y=[R[(n,L)][s] for n in have]
        c=[y[i]-(y[i-1]+y[i+1])/2 for i in range(1,len(y)-1)]
        med=st.median(c);mad=st.median(abs(x-med) for x in c);th=-max(.002,3*mad)
        hit=[(have[i+1],round(x,4)) for i,x in enumerate(c) if x<th]
        if hit: print(L,s,hit,'clean' if s in clean else 'out')
