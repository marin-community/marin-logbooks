import json,sys,statistics as st,wandb
d=sys.argv[1]
R=json.load(open(d+'/stage1_scores.json'))
S=dict(wandb.Api(timeout=60).run("marin-community/marin_moe/hero-merge-smoke").summary)
P='seq_len_4096'
R['step-390000']={k.split('/')[-2]:v for k,v in S.items() if k.startswith(f'eval_merge/step-390000/{P}/') and k.endswith('/loss') and k.count('/')>=4}
subs=[k for k in R['raw'] if not k.startswith('__')]
OUT=('ptb','wikitext_103','gab','twitterAAE')
isout=lambda n:any(n.startswith(z) for z in OUT)
UNC=('wikipedia_english','github_python','github_cpp','bbc_news','arxiv_physics','arxiv_computer_science','ao3_english')
isunc=lambda n:any(n.startswith(z) for z in UNC)
clean=[n for n in subs if not isout(n)]
print('subs',len(subs),'clean',len(clean),'unc',sum(isunc(n) for n in clean),'palclean',sum(not isunc(n) for n in clean))
print('raw clean macro',st.mean(R['raw'][n] for n in clean),'official',st.mean(R['raw'][n] for n in subs),'wandb macro',R['raw'].get('__macro'))
series=['step-385500','step-386000','step-386500','step-387000','step-387500','step-388000','step-388500','step-389000','step-389500','step-390000','raw']
print('--SCREEN dense raw')
flag={}
for n in subs:
    y=[R[s][n] for s in series]
    c=[y[i]-(y[i-1]+y[i+1])/2 for i in range(1,len(y)-1)]
    med=st.median(c);mad=st.median(abs(x-med) for x in c)
    th=-max(0.002,3*mad)
    hit=[(series[i+1],round(x,5)) for i,x in enumerate(c) if x<th]
    if hit: flag[n]=hit
    print(f'{n[:28]:28s} clean={not isout(n)} mad={mad:.5f} th={th:.5f} minc={min(c):.5f}',hit or '')
print('flag',flag)
print('clean jitter RMS (macro curv)')
yc=[st.mean(R[s][n] for n in clean) for s in series]
c=[yc[i]-(yc[i-1]+yc[i+1])/2 for i in range(1,len(yc)-1)];print(' rms',(sum(x*x for x in c)/len(c))**.5, [round(x,5) for x in yc])
def score(name,subset):
    D={n:R[name][n]-R['raw'][n] for n in subset}
    v=sorted(D.values())
    mac=st.mean(v);imp=sum(x<0 for x in v);worst=max(v)
    u=st.mean(D[n] for n in subset if isunc(n));p=st.mean(D[n] for n in subset if not isunc(n))
    top=min(D,key=D.get);dt=st.mean(x for n,x in D.items() if n!=top)
    return mac,imp,len(subset),u,p,worst,top,D[top],dt
print('--CANDIDATES vs raw (clean 19)')
c2=['raw+pending','raw+meanbias','U4','U11','B11a50','B11a25','B11a50+pendingbias','B11a50+reproject','W3']
for n in c2:
    m,i,N,u,p,w,t,tv,dt=score(n,clean)
    ok=m<=-0.001 and i>=14 and u<0 and p<0 and w<=0.003 and dt<0
    off=st.mean(R[n][s]-R['raw'][s] for s in subs); ptb=[R[n][s]-R['raw'][s] for s in subs if s.startswith('ptb')][0]
    print(f'{n:20s} d={m:+.5f} imp={i}/{N} unc={u:+.5f} pal={p:+.5f} worst={w:+.5f} top={t[:18]} {tv:+.5f} droptop={dt:+.5f} | off23={off:+.5f} ptb={ptb:+.5f} gate4k={ok}')
if flag:
    sc=[n for n in clean if n not in flag]
    print('--post-screen',len(sc))
    for n in c2:
        m,i,N,u,p,w,t,tv,dt=score(n,sc); print(f'{n:20s} d={m:+.5f} imp={i}/{N} unc={u:+.5f} pal={p:+.5f} worst={w:+.5f} droptop={dt:+.5f}')
# per-subset table for best
for n in ('B11a50','B11a25','U4'):
    print(n,{k[:14]:round(R[n][k]-R['raw'][k],5) for k in sorted(clean,key=lambda s:R[n][s]-R['raw'][s])})
print('raw series vs T clean:',[(s,round(st.mean(R[s][n]-R['raw'][n] for n in clean),5)) for s in series])
print('reproject effect',st.mean(R['B11a50+reproject'][n]-R['B11a50'][n] for n in clean),'pending on blend',st.mean(R['B11a50+pendingbias'][n]-R['B11a50'][n] for n in clean))
