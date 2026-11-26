import json,os,subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path

for name in ('DAYTONA_RL_API_KEY','DAYTONA_EVAL_API_KEY'):
 r=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret',name],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(name+' resolution failed')
 os.environ[name]=r.stdout.strip()
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ('organization_id','organizationId','org_id','created_by','createdBy','email')}
 if isinstance(value,list):return [clean(v) for v in value]
 return value
jobs=[('rl-usage',['shellbox-daytona-usage','--api-key-env','DAYTONA_RL_API_KEY','--cpu','2','--memory-gib','8','--disk-gib','8','--json']),('eval-usage',['shellbox-daytona-usage','--api-key-env','DAYTONA_EVAL_API_KEY','--cpu','2','--memory-gib','8','--disk-gib','8','--json']),('rl-snapshots',['shellbox-daytona-snapshots','--api-key-env','DAYTONA_RL_API_KEY','inventory','--json'])]
def run(job):
 name,args=job
 r=subprocess.run(['uv','run','--project','lib/shellbox','--frozen','--extra','daytona',*args],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(name+' inspection failed')
 value=clean(json.loads(r.stdout))
 path=Path('/tmp/mochi-loop-guards/daytona-'+name+'-2328-sanitized.json')
 path.write_text(json.dumps({'observed_at':datetime.now(timezone.utc).isoformat(),'data':value},indent=2)+'\n')
 return {'inspection':name,'saved_sanitized':str(path),'provider_mutations':False}
with ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(run,jobs):print(json.dumps(result),flush=True)
