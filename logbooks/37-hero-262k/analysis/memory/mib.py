import re,collections,sys
rows=[]
for l in open(sys.argv[1]):
    m=re.search(r'(\d\d:\d\d:\d\d\.\d+).*Allocated (\d+) at (0x[0-9a-f]+)',l)
    if m: rows.append((m[1],'A',int(m[2]),int(m[3],16))); continue
    m=re.search(r'(\d\d:\d\d:\d\d\.\d+).*Freed ptr: (0x[0-9a-f]+)',l)
    if m: rows.append((m[1],'F',0,int(m[2],16)))
A=102456760440
idx=[i for i,r in enumerate(rows) if r[1]=='A' and r[2]==A]
print('pre-train rows',idx[0])
pre=collections.Counter(r[2] for r in rows[:idx[0]] if r[1]=='A'); print('pre-train 1MiB allocs',pre[1048576],'total pre allocs',sum(pre.values()))
SZ=1048576
live={};k=-1;births=collections.Counter();frees=collections.Counter();lifes=[];
i_b={}
for i,r in enumerate(rows):
    if k+1<len(idx) and i==idx[k+1]: k+=1
    if r[1]=='A':
        live[r[3]]=(r[2],k,r[0])
        if r[2]==SZ: births[k]+=1
    else:
        v=live.pop(r[3],None)
        if v and v[0]==SZ:
            frees[k]+=1; lifes.append((k-v[1],v[1],k,r[0]))
print('births per step',[births[i] for i in range(max(births)+1)])
print('frees  per step',[frees[i] for i in range(max(frees)+1)] if frees else [])
print('life samples (life,born,died,t):',lifes[:10],lifes[-6:])
print('still live 1MiB by birth step (0=first train step incl init -1=pre):',sorted(collections.Counter(v[1] for v in live.values() if v[0]==SZ).items())[:50])
# time-in-step of births
for j in idx[6:9]:
    t=[(rows[n][0][3:],hex(rows[n][3])) for n in range(j,min(j+4000,len(rows))) if rows[n][1]=='A' and rows[n][2]==SZ]
    print('step',rows[j][0][3:],t[:8])
