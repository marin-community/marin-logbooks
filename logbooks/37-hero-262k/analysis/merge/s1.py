import wandb,json,statistics as st
a=wandb.Api(timeout=120)
r=a.run("marin-community/marin_moe/hero-merge-4k-step390251")
print(r.state,r.lastHistoryStep)
h=list(r.scan_history(page_size=1000))
P='seq_len_4096'
res={}
for row in h:
    for k,v in row.items():
        if k.startswith('eval_merge/') and not k.startswith('eval_merge/distance') and v is not None:
            parts=k.split('/')
            if parts[2]==P and k.endswith('/loss') and len(parts)>=5:
                res.setdefault(parts[1],{})[parts[-2]]=v
            elif parts[2]==P and k.endswith('macro_loss'):
                res.setdefault(parts[1],{})['__macro']=v
            elif parts[2]==P and k.endswith('total_time'):
                res.setdefault(parts[1],{})['__time']=v
json.dump(res,open('%s/stage1_scores.json'%__import__('sys').argv[1],'w'),indent=1)
print({k:len(v) for k,v in res.items()})
dist={}
for row in h:
    for k,v in row.items():
        if k.startswith('eval_merge/distance/') and v is not None:
            dist.setdefault(row['_step'],{})[k[len('eval_merge/distance/'):]]=v
json.dump(dist,open('%s/stage1_distance.json'%__import__('sys').argv[1],'w'))
print('distance entries',len(dist))
s=r.history(stream="system",pandas=False,samples=4000)
for k in ['system.gpu.%d.memoryAllocatedBytes'%i for i in range(4)]+['system.proc.memory.rssMB','system.memory_percent']:
    v=[x[k] for x in s if k in x and x[k] is not None]
    if v: print(k,round(max(v)/2**30,1) if 'Bytes' in k else max(v))
