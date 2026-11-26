import os,subprocess
program=r'''
import asyncio,json,os
from datetime import datetime,timezone
from pathlib import Path
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.snapshot import SnapshotClassification,classify_snapshot,list_snapshots,referenced_snapshots,snapshot_last_used
async def main():
    now=datetime.now(timezone.utc)
    async with DaytonaConnection(api_key=os.environ['DAYTONA_RL_API_KEY']).client_factory()() as client:
        references=await referenced_snapshots(client)
        inventory=await list_snapshots(client)
        candidates=[s for s in inventory if classify_snapshot(s,now=now)==SnapshotClassification.STALE and s.name not in references and s.mem==1 and s.disk==3]
        candidates=sorted(candidates,key=snapshot_last_used)[:6]
        rows=[{'name':s.name,'provider_size_raw':s.size,'size_units':'provider size field; not interpreted','cpu':s.cpu,'memory_gib':s.mem,'disk_gib':s.disk,'last_used_at':snapshot_last_used(s).isoformat(),'idle_seconds':(now-snapshot_last_used(s)).total_seconds(),'general':s.general,'sandbox_referenced':s.name in references,'state':s.state.value} for s in candidates]
        receipt={'account':'dedicated_rl','observed_at':now.isoformat(),'provider_mutations':False,'selected_count':len(rows),'selection':'oldest six stale non-general, unreferenced 1GiB/3GiB snapshots; protect all other profiles','queued_reference_review_complete':False,'candidates':rows}
        Path('/tmp/mochi-loop-guards/six-rl-snapshot-deletion-preview.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))
asyncio.run(main())
'''
r=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret','DAYTONA_RL_API_KEY'],capture_output=True,text=True)
if r.returncode:raise RuntimeError('RL credential resolution failed')
os.environ['DAYTONA_RL_API_KEY']=r.stdout.strip()
result=subprocess.run(['uv','run','--project','lib/shellbox','--frozen','--extra','daytona','python','-c',program],capture_output=True,text=True,timeout=120)
if result.returncode:raise RuntimeError('Read-only six-snapshot preview failed')
print(result.stdout)
