"""Fetch binned per-GPU FB_USED from the hub. usage: fetch_fb.py out.csv START END BIN_SECONDS [chunk_hours]"""
import sys,subprocess,json,datetime as dt,csv
out,st,en,binsec=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]); ch=float(sys.argv[5]) if len(sys.argv)>5 else 6
def ms(s): return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()*1000)
a,b=ms(st),ms(en); w=csv.writer(open(out,'w')); w.writerow(['bin_ms','node','gpu','min','max','n'])
step=int(ch*3600e3); tot=0
while a<b:
    e=min(a+step,b)
    q=f"""SELECT (timestamp_ms/{binsec*1000})*{binsec*1000} AS bin_ms, node_name, json_get(attributes_json,'gpu_index') AS gpu, min(value) mn, max(value) mx, count(*) n FROM "telemetry_v1.node_agent" WHERE name='gpu_memory_used_bytes' AND timestamp_ms >= {a} AND timestamp_ms < {e} GROUP BY 1,2,3"""
    r=subprocess.run(['uv','run','finelog','query','marin',q],capture_output=True,text=True)
    if r.returncode: print('ERR',a,r.stderr[-300:],flush=True); sys.exit(1)
    k=0
    for l in r.stdout.splitlines():
        if not l.startswith('{'): continue
        d=json.loads(l); w.writerow([d['bin_ms'],d['node_name'],d['gpu'],d['mn'],d['mx'],d['n']]); k+=1
    tot+=k; a=e; print(dt.datetime.utcfromtimestamp(a/1000),k,flush=True)
print('done',tot)
