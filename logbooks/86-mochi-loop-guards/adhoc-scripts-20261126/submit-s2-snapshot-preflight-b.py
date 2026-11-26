import os,subprocess
inner=r'''
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from shellbox.backends.daytona.snapshot import SnapshotRequest
configure_coreweave_s3()
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/prepared'
identities=json.loads((StoragePath(output)/'required-snapshots.json').read_text())
base=[sys.executable,'-m','shellbox.backends.daytona.snapshot_cli','--api-key-env','DAYTONA_RL_API_KEY']
before=subprocess.run([*base,'inventory','--json'],capture_output=True,text=True,timeout=120)
report={'gpu_submitted':False,'model_trials':0,'credential_present':bool(os.environ.get('DAYTONA_RL_API_KEY')),'inventory_exit_code':before.returncode,'snapshot_creation_attempted':False,'all_required_active':False}
if before.returncode:
    report.update(blocked_phase='inventory',provider_diagnostic=redact_string(before.stdout+'\n'+before.stderr)[-6000:])
else:
    prior={r['name']:r for r in json.loads(before.stdout)}
    report['required_present_before']=sorted(set(prior)&{r['name'] for r in identities})
    target=Path('/tmp/s2-restored-contexts');target.mkdir()
    requests=[]
    for identity in identities:
        archive=target/(identity['name']+'.tar.gz')
        archive.write_bytes(StoragePath(identity['context_archive']).read_bytes())
        with tarfile.open(archive) as bundle:bundle.extractall(target,filter='data')
        dockerfile=target/identity['name']/'Dockerfile'
        assert hashlib.sha256(dockerfile.read_bytes()).hexdigest()==identity['dockerfile_sha256']
        spec={'name':identity['name'],'resources':identity['resources'],'dockerfile':str(dockerfile)}
        request=SnapshotRequest.from_dict(spec)
        assert request.name==identity['name']
        requests.append(spec)
    manifest=target/'required-specs.json';manifest.write_text(json.dumps(requests))
    ensure=subprocess.run([*base,'ensure','--specs',str(manifest)],capture_output=True,text=True,timeout=2700)
    report.update(snapshot_creation_attempted=True,ensure_exit_code=ensure.returncode,provider_diagnostic=redact_string(ensure.stdout+'\n'+ensure.stderr)[-6000:])
    after=subprocess.run([*base,'inventory','--json'],capture_output=True,text=True,timeout=120)
    report['post_inventory_exit_code']=after.returncode
    if not after.returncode:
        following={r['name']:r for r in json.loads(after.stdout)}
        states={r['name']:following.get(r['name'],{}).get('state') for r in identities}
        report.update(required_states=states,all_required_active=all(state=='active' for state in states.values()))
    else:report['post_inventory_diagnostic']=redact_string(after.stdout+'\n'+after.stderr)[-2000:]
(StoragePath(output)/'snapshot-preflight-b.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
'''
outer='\n'.join([
'import os,subprocess',
'from marin.rl.skyrl import skyrl_cpu_command',
'from rigging.secrets import require_secret',
"os.environ['DAYTONA_RL_API_KEY']=require_secret('DAYTONA_RL_API_KEY')",
f"subprocess.run(skyrl_cpu_command('python','-c',{inner!r}),check=True)",
])
compile(inner,'snapshot-preflight-b','exec');compile(outer,'snapshot-preflight-b-driver','exec')
credential=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret','DAYTONA_RL_API_KEY'],capture_output=True,text=True)
if credential.returncode:raise RuntimeError('RL secret resolution failed')
os.environ['DAYTONA_RL_API_KEY']=credential.stdout.strip()
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','4','--memory','16GB','--disk','64GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-s2-snapshot-preflight-20261125b','--secret','DAYTONA_RL_API_KEY','env:DAYTONA_RL_API_KEY','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(args).returncode)
