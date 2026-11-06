import os,json,wandb,pandas as pd,numpy as np,sys
api=wandb.Api(timeout=120)
jn=json.load(open(sys.argv[1]))
d=pd.read_csv(sys.argv[2]);d['v']=d['max']/2**30;d['g']=d.node+':'+d.gpu.astype(str)
piv=d.pivot_table(index='bin_ms',columns='g',values='v')
rows=[]
for j,info in jn.items():
    name=j.split('grug-train-')[-1]
    if 'mrcr' in name: continue
    if not info['attempts']: continue
    try: r=api.run(f"marin-community/marin_moe/{name}")
    except Exception as e: continue
    h=[x for x in r.scan_history(keys=['_step','_timestamp','throughput/mfu'],page_size=500)]
    if len(h)<10: continue
    t0=int(h[0]['_timestamp']*1000);t1=int(h[-1]['_timestamp']*1000);steps=len(h)
    a=max(info['attempts'],key=int);nodes=set(info['attempts'][a])
    cols=[c for c in piv.columns if c.split(':')[0] in nodes]
    x=piv.loc[t0-60000:t1,cols]
    if len(x)<3: continue
    # sum of small positive steps per GPU
    df=x.ffill().diff()
    small=df.where((df>0.01)&(df<0.3)).sum()
    big=df.where(df>=0.3).sum()
    neg=df.where(df<-0.01).sum()
    sm=small*1024/steps
    lv=x.iloc[0]
    rows.append(dict(name=name,gpus=len(cols),steps=steps,mins=round((t1-t0)/60000),s_step=round((t1-t0)/1000/steps,1),lvl0=round(lv.median(),2),lvlmax=round(x.max().median(),2),MiB_per_step=round(sm.mean(),3),p90=round(sm.quantile(.9),3),max=round(sm.max(),3),bigGiB=round(big.mean(),2),negGiB=round(neg.mean(),2)))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(sys.argv[3],index=False)
