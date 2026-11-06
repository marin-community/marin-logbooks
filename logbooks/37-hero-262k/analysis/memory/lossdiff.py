import os,sys,statistics as st,wandb
api=wandb.Api(timeout=120)
R={}
for rid in ['memgrow-vlog','memgrow-thresh','memgrow-buf16','memgrow-buf48']:
    r=api.run(f"marin-community/marin_moe/{rid}")
    h={x['_step']:x for x in r.scan_history(keys=['_step','train/loss','throughput/duration','moe/drop_fraction','throughput/loading_time'],page_size=200)}
    R[rid]=h
    d=[x['throughput/duration'] for s,x in h.items() if s>390252]
    print(rid,r.state,'n',len(h),'dur med %.2f mean %.2f max %.1f'%(st.median(d),st.mean(d),max(d)),'drop med %.3e'%st.median(x['moe/drop_fraction'] for x in h.values()),'load max %.2f'%max(x['throughput/loading_time'] for x in h.values()),flush=True)
b=R['memgrow-vlog']
for rid,h in R.items():
    c=[s for s in h if s in b]
    print(rid,'steps common',len(c),'max|dloss| %.2e'%max(abs(h[s]['train/loss']-b[s]['train/loss']) for s in c),'first',[ (s,round(h[s]['train/loss'],5)) for s in sorted(c)[:2]],flush=True)
