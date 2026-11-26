import os,subprocess
inner=r'''
import asyncio,hashlib,json,os,tarfile
from pathlib import Path
from daytona import DaytonaNotFoundError
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.machine import DaytonaImageBuild,DaytonaMachineFactory
from shellbox.backends.daytona.snapshot import SnapshotRequest
from shellbox.machine import MachineSpec,NetworkPolicy
configure_coreweave_s3()
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/prepared'
identities=json.loads((StoragePath(output)/'required-snapshots.json').read_text())
metadata=json.loads((StoragePath(output)/'receipt.json').read_text())
target=Path('/tmp/s2-declarative-contexts');target.mkdir()
connection=DaytonaConnection(api_key=os.environ['DAYTONA_RL_API_KEY'])
async def main():
    results=[]
    for identity in identities:
        archive=target/(identity['name']+'.tar.gz');archive.write_bytes(StoragePath(identity['context_archive']).read_bytes())
        with tarfile.open(archive) as bundle:bundle.extractall(target,filter='data')
        dockerfile=target/identity['name']/'Dockerfile'
        assert hashlib.sha256(dockerfile.read_bytes()).hexdigest()==identity['dockerfile_sha256']
        request=SnapshotRequest.from_dict({'name':identity['name'],'resources':identity['resources'],'dockerfile':str(dockerfile)})
        runtime=metadata['runtimes'][identity['name']]
        spec=MachineSpec(source=request.source,workdir='',network=NetworkPolicy.ALLOW,cpus=runtime['cpus'],memory_mb=runtime['memory_mb'],storage_mb=runtime['storage_mb'],gpus=runtime['gpus'],startup_timeout=runtime['startup_timeout'])
        factory=DaytonaMachineFactory(connection.client_factory(),ttl_minutes=480,labels={'experiment':'mochi-loop-guards','purpose':'s2-declarative-preflight'},image_build=DaytonaImageBuild.DECLARATIVE)
        result={'snapshot_identity':identity['name'],'exact_image_and_resources_preserved':True,'model_trials':0,'created':False}
        machine=None
        try:
            machine=await factory.create(spec)
            result.update(created=True,sandbox_id=machine.sandbox.id)
        except Exception as error:
            result.update(error_type=type(error).__name__,diagnostic=redact_string(str(error))[:4000])
        finally:
            if machine is not None:
                await machine.close()
                async with connection.client_factory()() as client:
                    try:await client.get(machine.sandbox.id)
                    except DaytonaNotFoundError:result['owned_sandbox_absent_after_close']=True
                    else:result['owned_sandbox_absent_after_close']=False
        results.append(result)
        receipt={'gpu_submitted':False,'model_trials':0,'account':'dedicated_rl','mode':'declarative','results':results,'all_four_supported':len(results)==4 and all(r['created'] and r.get('owned_sandbox_absent_after_close') for r in results)}
        (StoragePath(output)/'declarative-preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not result['created']:break
asyncio.run(main())
'''
outer='\n'.join(['import os,subprocess','from marin.rl.skyrl import skyrl_cpu_command','from rigging.secrets import require_secret',"os.environ['DAYTONA_RL_API_KEY']=require_secret('DAYTONA_RL_API_KEY')",f"subprocess.run(skyrl_cpu_command('python','-c',{inner!r}),check=True)"])
compile(inner,'s2-declarative-preflight','exec');compile(outer,'s2-declarative-preflight-driver','exec')
credential=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret','DAYTONA_RL_API_KEY'],capture_output=True,text=True)
if credential.returncode:raise RuntimeError('RL secret resolution failed')
os.environ['DAYTONA_RL_API_KEY']=credential.stdout.strip()
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','4','--memory','16GB','--disk','64GB','--enable-extra-resources','--timeout','9000','--sync-package','marin-core','--job-name','mochi-loop-guards-s2-declarative-preflight-20261125a','--secret','DAYTONA_RL_API_KEY','env:DAYTONA_RL_API_KEY','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(args).returncode)
