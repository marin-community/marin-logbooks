from pathlib import Path
source=Path('/tmp/mochi-loop-guards/submit-s2-environment-census.py').read_text()
source=source.replace('import subprocess\nfrom pathlib', 'import os,subprocess\nfrom pathlib',1)
source=source.replace('import hashlib,json,sys\n','import hashlib,json,sys,shutil,subprocess,tarfile\n',1)
source=source.replace('from rigging.filesystem.storage_path import StoragePath\n','from rigging.filesystem.storage_path import StoragePath\nfrom rigging.filesystem.transfer import copy\nfrom rigging.redaction import redact_string\n',1)
source=source.replace('counts=Counter();families=Counter();requests={};roles=Counter();runtimes={}','counts=Counter();families=Counter();requests={};roles=Counter();runtimes={};native_specs={}',1)
old="""                    payload=request.as_dict()
                    if 'dockerfile' in payload:
                        payload['source_kind']='dockerfile'
                        payload['dockerfile_sha256']=hashlib.sha256(Path(payload.pop('dockerfile')).read_bytes()).hexdigest()
                    else:
                        payload['source_kind']='registry_image'
                        assert '@sha256:' in payload['image']
"""
new="""                    native=request.as_dict()
                    if request.name not in native_specs:
                        if 'dockerfile' in native:
                            context=Path('/tmp/s2-required-contexts')/request.name
                            shutil.copytree(machine.source.context,context)
                            native['dockerfile']=str(context/Path(machine.source.dockerfile).relative_to(machine.source.context))
                            archive=context.with_suffix('.tar.gz')
                            with tarfile.open(archive,'w:gz') as bundle:bundle.add(context,arcname=request.name)
                            copy(str(archive),str(StoragePath(output)/'contexts'/archive.name))
                        native_specs[request.name]=native
                    payload=dict(native_specs[request.name])
                    if 'dockerfile' in payload:
                        payload['source_kind']='dockerfile'
                        payload['dockerfile_sha256']=hashlib.sha256(Path(payload.pop('dockerfile')).read_bytes()).hexdigest()
                        payload['context_archive']=str(StoragePath(output)/'contexts'/(request.name+'.tar.gz'))
                    else:
                        payload['source_kind']='registry_image'
                        assert '@sha256:' in payload['image']
"""
assert source.count(old)==1
source=source.replace(old,new,1)
marker="print(json.dumps({k:v for k,v in result.items() if k not in ('request_counts','runtimes')}))\n'''"
append=r'''print(json.dumps({k:v for k,v in result.items() if k not in ('request_counts','runtimes')}))
Path('/tmp/s2-required-specs.json').write_text(json.dumps(list(native_specs.values())))
base=[sys.executable,'-m','shellbox.backends.daytona.snapshot_cli','--api-key-env','DAYTONA_RL_API_KEY']
before=subprocess.run([*base,'inventory','--json'],capture_output=True,text=True,timeout=120)
if before.returncode:raise RuntimeError('RL snapshot inventory failed before ensure')
prior=json.loads(before.stdout)
prior_names={r['name'] for r in prior}
ensure=subprocess.run([*base,'ensure','--specs','/tmp/s2-required-specs.json'],capture_output=True,text=True,timeout=2700)
after=subprocess.run([*base,'inventory','--json'],capture_output=True,text=True,timeout=120)
if after.returncode:raise RuntimeError('RL snapshot inventory failed after ensure')
following={r['name']:r for r in json.loads(after.stdout)}
ready={name:following.get(name,{}).get('state') for name in native_specs}
preflight={'preflight_completed':True,'gpu_submitted':False,'model_trials':0,'snapshot_creation_attempted':True,'ensure_exit_code':ensure.returncode,'all_required_active':all(state=='active' for state in ready.values()),'required_states':ready,'required_present_before':sorted(set(native_specs)&prior_names),'provider_diagnostic':redact_string(ensure.stdout+'\n'+ensure.stderr)[-4000:]}
(StoragePath(output)/'snapshot-preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
print(json.dumps(preflight))
'''
assert source.count(marker)==1
source=source.replace(marker,append+"'''",1)
source=source.replace('/s2-environments','/s2-environments/prepared')
source=source.replace("'from pathlib import Path',", "'from pathlib import Path',\n'from rigging.secrets import require_secret',",1)
source=source.replace("'configure_coreweave_s3()',", "'configure_coreweave_s3()',\n\"os.environ['DAYTONA_RL_API_KEY']=require_secret('DAYTONA_RL_API_KEY')\",",1)
source=source.replace('mochi-loop-guards-s2-environment-census-20261125b','mochi-loop-guards-s2-snapshot-preflight-20261125a')
source=source.replace("'--job-name','mochi-loop-guards-s2-snapshot-preflight-20261125a','--'", "'--job-name','mochi-loop-guards-s2-snapshot-preflight-20261125a','--secret','DAYTONA_RL_API_KEY','env:DAYTONA_RL_API_KEY','--'")
source=source.replace("raise SystemExit(subprocess.run(args).returncode)", "credential=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret','DAYTONA_RL_API_KEY'],capture_output=True,text=True)\nif credential.returncode:raise RuntimeError('RL secret resolution failed')\nos.environ['DAYTONA_RL_API_KEY']=credential.stdout.strip()\nraise SystemExit(subprocess.run(args).returncode)")
compile(source,'s2-snapshot-preflight','exec')
Path('/tmp/mochi-loop-guards/submit-s2-snapshot-preflight.py').write_text(source)
print('Prepared CPU-only exact-identity snapshot preflight; no job submitted by this preparation script.')
