import os,subprocess
from pathlib import Path
program=r'''
import asyncio,json,os
from datetime import datetime,timezone
from pathlib import Path
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.snapshot import SnapshotClassification,classify_snapshot,list_snapshots,referenced_snapshots,snapshot_last_used
configure_coreweave_s3()
plan=json.loads(Path('/tmp/mochi-loop-guards/six-rl-snapshot-deletion-preview.json').read_text())
assert plan['queued_reference_review_complete'] and plan['selected_count']==6
report={**plan,'previous_attempt':{'provider_mutations':False,'failure':'Helper supplied a name to SDK delete, which requires Snapshot; all six rechecked active before retry'},'provider_mutations':False,'started_at':datetime.now(timezone.utc).isoformat(),'deletions':[]}
local=Path('/tmp/mochi-loop-guards/six-rl-snapshot-reclamation-receipt.json')
remote=StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/six-snapshot-reclamation.json')
def save():
    body=json.dumps(report,indent=2)+'\n';local.write_text(body);remote.write_text(body)
async def main():
    async with DaytonaConnection(api_key=os.environ['DAYTONA_RL_API_KEY']).client_factory()() as client:
        for target in plan['candidates']:
            now=datetime.now(timezone.utc)
            references=await referenced_snapshots(client)
            snapshot=await client.snapshot.get(target['name'])
            last=snapshot_last_used(snapshot)
            if classify_snapshot(snapshot,now=now)!=SnapshotClassification.STALE or snapshot.name in references or snapshot.mem!=1 or snapshot.disk!=3 or last is None:
                report['refused_target']=target['name'];save();raise RuntimeError('Snapshot reference/profile/last-use changed; cleanup stopped')
            row={**target,'revalidated_at':now.isoformat(),'last_used_at':last.isoformat(),'idle_seconds':(now-last).total_seconds(),'provider_size_raw':snapshot.size,'deletion_requested':False}
            report['deletions'].append(row);save()
            await client.snapshot.delete(snapshot)
            report['provider_mutations']=True;row['deletion_requested']=True;row['request_accepted_at']=datetime.now(timezone.utc).isoformat();save()
        following={s.name:s.state.value for s in await list_snapshots(client)}
        for row in report['deletions']:
            row['absent_after_delete']=row['name'] not in following
            row['remaining_state']=following.get(row['name'])
        report.update(finished_at=datetime.now(timezone.utc).isoformat(),accepted_count=sum(r['deletion_requested'] for r in report['deletions']),absent_count=sum(r['absent_after_delete'] for r in report['deletions']))
        save()
        print(json.dumps({'accepted_count':report['accepted_count'],'absent_count':report['absent_count'],'receipt':str(remote)}))
asyncio.run(main())
'''
credential=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret','DAYTONA_RL_API_KEY'],capture_output=True,text=True)
if credential.returncode:raise RuntimeError('RL credential resolution failed')
os.environ['DAYTONA_RL_API_KEY']=credential.stdout.strip()
result=subprocess.run(['uv','run','--project','lib/shellbox','--frozen','--extra','daytona','python','-c',program],capture_output=True,text=True,timeout=180)
Path('/tmp/mochi-loop-guards/reclamation-native-stderr.log').write_text(result.stderr)
Path('/tmp/mochi-loop-guards/reclamation-native-stdout.log').write_text(result.stdout)
if result.returncode:raise RuntimeError('Bounded reclamation did not complete; inspect durable receipt before any retry')
print(result.stdout)
