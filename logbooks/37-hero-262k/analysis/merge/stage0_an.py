import numpy as np, sys, itertools, json
d=sys.argv[1]
S=[378000,384000,385500,386000,386500,387000,387500,388000,388500,389000,389500,390000,390251]
P={};B={}
for s in S:
    z=np.load(f"{d}/{s}.npz"); P[s]=z["bias"].astype(np.float64); b=-z["pending"].astype(np.float64); B[s]=b-b.mean(-1,keepdims=True)
fro=lambda x:np.linalg.norm(x)
rms=lambda x:np.sqrt(np.mean(x**2))
print("P col-mean max",max(np.abs(P[s].mean(-1)).max() for s in S)," pending raw mean", np.mean([ (-np.load(f'{d}/{s}.npz')['pending']).mean() for s in S]))
print("step: rmsP rmsB rms(P-B) rel=|P-B|/|B| maxabs(P-B) | corr(P,B)")
gap1=[]
for s in S:
    x=P[s]-B[s]; gap1.append(rms(x)); print(s,f"{rms(P[s]):.4f} {rms(B[s]):.4f} {rms(x):.4f} {fro(x)/fro(B[s]):.3f} {np.abs(x).max():.3f} {np.corrcoef(P[s].ravel(),B[s].ravel())[0,1]:.4f}")
print("gap1 mean rms",np.mean(gap1))
# per layer gap1
L=np.mean([np.sqrt(((P[s]-B[s])**2).mean(-1)) for s in S],0); Lb=np.mean([np.sqrt((B[s]**2).mean(-1)) for s in S],0)
print("per-layer gap1 rms:",np.round(L,3).tolist()); print("per-layer bias rms:",np.round(Lb,3).tolist())
# structure function
rows=[]
for a,b in itertools.combinations(S,2):
    rows.append((b-a,'PP',rms(P[a]-P[b]))); rows.append((b-a,'BB',rms(B[a]-B[b])))
    # cross with gap: B[a] from batch a; P[b] from batch b-1
    rows.append((b-a-1,'BP',rms(B[a]-P[b]))); 
    if b-a>1: rows.append((b-a+1,'PB',rms(P[a]-B[b])))
for s in S: rows.append((1,'P_B',rms(P[s]-B[s])))
gaps=sorted(set(r[0] for r in rows))
print("gap | n | D_BB D_PP D_cross(mean)")
bins=[(1,1),(250,252),(499,501),(749,752),(999,1001),(1250,1252),(1499,1501),(1999,2001),(2500,2502),(2999,3001),(3500,4001),(4000,4501),(4501,6500),(6500,8000),(8000,10000),(10000,12252)]
out=[]
for lo,hi in bins:
    v=[r for r in rows if lo<=r[0]<=hi]
    if not v: continue
    def m(t):
        x=[r[2] for r in v if r[1] in t]; return (np.mean(x) if x else np.nan,len(x))
    print(f"{lo}-{hi}: BB {m(['BB'])[0]:.4f}(n{m(['BB'])[1]}) PP {m(['PP'])[0]:.4f} cross {m(['BP','PB','P_B'])[0]:.4f}(n{m(['BP','PB','P_B'])[1]})")
    out.append((np.mean([r[0] for r in v]),np.mean([r[2]**2 for r in v]),len(v)))
g=np.array([r[0] for r in rows],float); y=np.array([r[2]**2 for r in rows])
# fit D2 = c + k*g on all
A=np.c_[np.ones_like(g),g]; c=np.linalg.lstsq(A,y,rcond=None)[0]; print("fit D2=c+k g: c",c[0],"k/500",c[1]*500,"noise rms per draw",np.sqrt(c[0]/2))
print("drift at 251 D2",c[1]*251," frac of total",c[1]*251/(c[0]+c[1]*251)); print("gap at which drift=noise (steps)",c[0]/c[1])
yp=np.c_[np.ones_like(g),g,g**2]; c2=np.linalg.lstsq(yp,y,rcond=None)[0]; print("quad fit",c2)
# drift vs step (excluding very long): dense only
dense=[r for r in rows if r[0]<=5000]; g2=np.array([r[0] for r in dense],float);y2=np.array([r[2]**2 for r in dense]);c3=np.linalg.lstsq(np.c_[np.ones_like(g2),g2],y2,rcond=None)[0];print("dense fit c,k/500",c3[0],c3[1]*500)
# independence: lag-based covariance of residuals around a linear trend per-element
win=S[2:]
Pw=np.stack([P[s] for s in win]);Bw=np.stack([B[s] for s in win])
t=np.array(win,float);t=(t-t.mean())
def resid(X):
    M=np.c_[np.ones_like(t),t];coef=np.linalg.lstsq(M,X.reshape(len(t),-1),rcond=None)[0];return (X.reshape(len(t),-1)-M@coef).reshape(X.shape),coef
Rb,cb=resid(Bw)
print("B residual rms around window linear trend",rms(Rb)*np.sqrt(len(t)/(len(t)-2)))
print("B slope rms per 500 steps",rms(cb[1])*500)
mean11=Bw.mean(0); meanP=Pw.mean(0)
print("rms(B_T - mean11)",rms(B[390251]-mean11)," rms(P_T-mean11)",rms(P[390251]-mean11)," rms(meanB-meanP)",rms(mean11-meanP), "expected se of mean11 under iid", np.mean(gap1)/np.sqrt(2)/np.sqrt(11))
# structural: variance across experts of cross-checkpoint noise, heavy tails
x=np.stack([P[s]-B[s] for s in S]); print("abs(P-B) pct 50,90,99,99.9",np.percentile(np.abs(x),[50,90,99,99.9]).round(3).tolist())
print("per-expert persistence: corr of (P-B) between checkpoints mean",np.mean([np.corrcoef(x[i].ravel(),x[j].ravel())[0,1] for i,j in itertools.combinations(range(13),2)]))
#per layer type: global_every
print("corr P_T vs earliest",np.corrcoef(B[378000].ravel(),B[390251].ravel())[0,1])
