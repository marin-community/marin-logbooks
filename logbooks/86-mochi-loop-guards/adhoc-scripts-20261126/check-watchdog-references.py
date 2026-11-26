import json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
registry=yaml.safe_load(Path('/Users/benfeuer/Documents/experiments/active/job-watchdog/registry.yaml').read_text())
entries=[e for e in registry['entries'] if e.get('iris')]
def check(e):
 iris=e['iris']
 r=subprocess.run(['uv','run','--no-sync','iris','--cluster',iris['cluster'],'job','describe',iris['job']],capture_output=True,text=True,timeout=60)
 Path('/tmp/mochi-loop-guards/snapshot-reference-'+e['id']+'.log').write_text(r.stdout+r.stderr)
 return {'id':e['id'],'job_id':iris['job'],'cluster':iris['cluster'],'query_exit':r.returncode,'state':next((l for l in r.stdout.splitlines() if l.startswith('State:')),None),'tasks':next((l for l in r.stdout.splitlines() if l.startswith('Tasks:')),None)}
with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(check,entries))
Path('/tmp/mochi-loop-guards/snapshot-watchdog-reference-states.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='job_id'} for r in results]))
