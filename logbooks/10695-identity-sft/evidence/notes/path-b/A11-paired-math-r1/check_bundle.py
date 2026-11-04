from pathlib import Path
from unittest.mock import patch
import hashlib,inspect,io,json,os,subprocess,sys,tempfile,zipfile
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from experiments.post_training.identity_fullscreen_owner import math_only,owner
HERE=Path(__file__).resolve().parent;plan=json.loads((HERE/'launch-plan.json').read_bytes());root=Path(plan['workspace']);parent=HERE.parent/'A8a-paired-math-r2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent/'receipt.json')==plan['inherited_packet']['sha256']
for item in json.loads((parent/'receipt.json').read_bytes())['files']:assert sha(Path(item['path']))==item['sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
changes=subprocess.check_output(['git','diff','--name-only',plan['parent'],plan['code_pin']],cwd=root,text=True).splitlines();target=plan['targets']['A11'];assert changes==[target['committed_path']]
ready_path=Path(plan['actual_readiness']['path']);assert sha(ready_path)==plan['actual_readiness']['sha256']
ready=json.loads(ready_path.read_bytes());assert ready['HF']==target['candidate']['model_path']
for entry in ready['metadata'].values():assert Path(entry['path']).stat().st_size==entry['bytes'] and sha(Path(entry['path']))==entry['sha256']
config=Path(plan['actual_HF_config']['path']);assert sha(config)==plan['actual_HF_config']['sha256']
assert sha(Path(plan['actual_HF_config']['receipt']))==plan['actual_HF_config']['receipt_sha256']
guards={x['path']:x['sha256'] for x in target['candidate']['model_guard']}
for key in ['audit.json','event.json','model.safetensors.index.json']:
 entry=ready['metadata'][key];assert guards[entry['uri']]==entry['sha256']
assert guards[ready['HF']+'/config.json']==sha(config)
old=json.loads((parent/'A8a-candidate.json').read_bytes());new=target['candidate'];assert new==json.loads((root/target['committed_path']).read_bytes())
oldargv=old['screen']['serve_argv'];newargv=new['screen']['serve_argv'];assert [x for i,x in enumerate(oldargv) if i not in [2,oldargv.index('--name')+1,oldargv.index('--endpoint-name')+1]]==[x for i,x in enumerate(newargv) if i not in [2,newargv.index('--name')+1,newargv.index('--endpoint-name')+1]]
for a,b in zip(old['math'],new['math'],strict=True):
 assert a['trial']==b['trial']
 for flag in ['--dtype','--tokenizer','--sample-seed']:
  if flag in a['argv']:assert a['argv'][a['argv'].index(flag)+1]==b['argv'][b['argv'].index(flag)+1]
archive=Path(plan['source_archive']['path']).read_bytes();native=create_workspace_zip(root);checks=[]
with tempfile.TemporaryDirectory(prefix='a11math-native-') as directory,zipfile.ZipFile(io.BytesIO(native)) as z:
 dest=Path(directory);z.extractall(dest);env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONSAFEPATH='1',IRIS_PYTHON=sys.executable,IRIS_VENV=sys.prefix,IRIS_WORKDIR=str(dest));env['PYTHONPATH']=':'.join([str(dest),*[str(p) for p in (dest/'lib').glob('*/src')]])
 assert z.read(target['committed_path'])==(root/target['committed_path']).read_bytes()
 req=math_only.submission(name=target['root_name'],candidate=new,archive=archive,credentials={'CW_KEY_ID':'fixture','CW_KEY_SECRET':'fixture'});inspect.signature(IrisClient.submit).bind(None,**req)
 assert req['entrypoint'].workdir_files=={math_only.ARCHIVE_PATH:archive};assert req['timeout'].to_seconds()==49*3600;assert any(any(v.value=='cw-rno2a' for v in c.values) for c in req['constraints'])
 cmd=req['entrypoint'].command.copy();cmd[-1]+=' --inspect-only';actual=subprocess.run(cmd,cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90);assert actual.stdout.strip().splitlines()[-1]==new['label']
 for trial in ['greedy','17','29']:
  with patch.dict(os.environ,IRIS_WORKDIR=str(dest)):entry=owner.task_entrypoint(new,'math',trial)
  cmd=entry.command.copy();cmd[-1]+=' --inspect-only';actual=subprocess.run(cmd,cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90);view=json.loads(actual.stdout.strip().splitlines()[-1]);assert view['candidate_sha256']==target['candidate_sha256'];assert view['command_argv']==next(x['argv'] for x in new['math'] if x['trial']==trial);checks.append(view)
proof=dict(code_pin=plan['code_pin'],native_bundle_sha256=hashlib.sha256(native).hexdigest(),native_bundle_bytes=len(native),source_delta=changes,all_parent_owner_f5_driver_science_bytes_unchanged=True,one_root_three_actual_extracted_children=checks,new_reference_serve=False,reference=plan['reused_reference'],per_benchmark_modes=plan['method'],remote_calls=0)
(HERE/'native-binding-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='one_root_three_actual_extracted_children'},indent=2))
