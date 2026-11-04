from pathlib import Path
import hashlib,inspect,io,json,os,subprocess,sys,tempfile,zipfile
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from experiments.post_training.identity_path_b import hf_archive
HERE=Path(__file__).resolve().parent;plan=json.loads((HERE/'launch-plan.json').read_bytes());root=Path(plan['workspace']);sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
config=json.loads((hf_archive.ARTIFACTS/'targets.json').read_bytes());assert [x['label'] for x in config['targets']]==['A3','A8a']
for target in config['targets']:
 assert target['repo_id']==f"open-athena/Snowball-67B-A2B-Identity-PathB-{target['label']}-2026.10.30"
 manifest=json.loads(hf_archive.checked_artifact(target['manifest_file'],target['manifest_sha256']));assert manifest['destination']==target['source'];assert len(manifest['objects'])==56;assert sum(x['is_indexed_shard'] for x in manifest['objects'])==39
 assert all(x['source_sha256']==x['destination_sha256'] for x in manifest['objects'])
 card=hf_archive.checked_artifact(target['card_file'],target['card_sha256']).decode();assert card.count(hf_archive.WARNING)==card.count(hf_archive.RESPONSIBLE_USE)==1;assert 'license: openmdw-1.1' in card
req=hf_archive.submission({'HF_TOKEN':'fixture','CW_KEY_ID':'fixture','CW_KEY_SECRET':'fixture'});inspect.signature(IrisClient.submit).bind(None,**req)
fn,args,kwargs=req['entrypoint'].resolve();assert fn is hf_archive.run and not args and not kwargs
assert set(req['environment'].secrets)=={'HF_TOKEN','CW_KEY_ID','CW_KEY_SECRET'};assert not req['environment'].env_vars
assert any(any(v.value=='cw-rno2a' for v in c.values) for c in req['constraints']);assert req['resources'].cpu==8 and req['timeout'].to_seconds()==12*3600;assert req['max_retries_preemption']==req['max_retries_failure']==0
native=create_workspace_zip(root)
with zipfile.ZipFile(io.BytesIO(native)) as z,tempfile.TemporaryDirectory(prefix='private-hf-native-') as directory:
 files=[p for p in hf_archive.ARTIFACTS.iterdir() if p.is_file()]+[root/'experiments/post_training/identity_path_b/hf_archive.py'];assert len(files)==9
 for p in files:assert z.read(str(p.relative_to(root)))==p.read_bytes()
 dest=Path(directory);z.extractall(dest);env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONSAFEPATH='1');env['PYTHONPATH']=':'.join([str(dest),'/mnt/storage/karan-scratch/genet-path-b/archive-python-deps',*[str(p) for p in (dest/'lib').glob('*/src')]])
 completed=subprocess.run([sys.executable,'-P','-m','experiments.post_training.identity_path_b.hf_archive','--inspect-only'],cwd=dest,env=env,text=True,capture_output=True,check=True,timeout=90);assert json.loads(completed.stdout)==config
proof=dict(code_pin=plan['code_pin'],native_bundle_sha256=sha(native),native_bundle_bytes=len(native),members=len(z.namelist()),committed_worker_and8artifacts_exact=True,extracted_actual_module_CLI='PASS',native_CPU_request_signature='PASS',explicit_RNO=True,CPU=8,requested_hours=12,secrets_names_only=sorted(req['environment'].secrets),delete_operations=0,source_objects_per_model=56,indexed_shards_per_model=39,model_or_storage_or_Hub_calls=0)
(HERE/'native-binding-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
