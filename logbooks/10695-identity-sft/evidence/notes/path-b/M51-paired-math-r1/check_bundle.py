from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast,hashlib,inspect,io,json,os,subprocess,sys,tempfile,zipfile
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from iris.cluster.types import JobName
from experiments.post_training.identity_fullscreen_owner import math_only,owner
H=Path(__file__).resolve().parent;B=H.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def functions(source):
 tree=ast.parse(source)
 for n in ast.walk(tree):
  if isinstance(n,ast.Attribute)and n.attr=='PRIORITY_BAND_INTERACTIVE':n.attr='PRIORITY_BAND_PRODUCTION'
 return {n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def main():
 p=json.loads((H/'launch-plan.json').read_text());W=Path(p['workspace']);label=next(iter(p['targets']));t=p['targets'][label];new=t['candidate'];assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==p['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
 for file in ['owner.py','math_only.py']:
  rel='experiments/post_training/identity_fullscreen_owner/'+file;oldsrc=subprocess.check_output(['git','show',p['parent']+':'+rel],cwd=W,text=True);assert functions(oldsrc)==functions((W/rel).read_text())
 for file in ['candidate.py','frozen/run_math.py','math_signature.py','persist_task.py']:
  rel='experiments/post_training/identity_fullscreen_owner/'+file;assert subprocess.check_output(['git','show',p['parent']+':'+rel],cwd=W)==(W/rel).read_bytes()
 rel='lib/marin/src/marin/inference/iris_cli.py';assert subprocess.check_output(['git','show',p['parent']+':'+rel],cwd=W)==(W/rel).read_bytes()
 old=json.loads((B/'A11-paired-math-r1/A11-candidate.json').read_text());a=list(old['screen']['serve_argv']);b=list(new['screen']['serve_argv']);a[2]=b[2]
 for flag in ['--name','--endpoint-name','--priority-band']:a[a.index(flag)+1]=b[b.index(flag)+1]
 assert a==b and b[b.index('--priority-band')+1]=='production'
 for a,b in zip(old['math'],new['math'],strict=True):
  x=list(a['argv']);y=list(b['argv'])
  for flag in ['--model','--output','--model-revision']:x[x.index(flag)+1]=y[y.index(flag)+1]
  assert x==y and a['trial']==b['trial']
 R=Path(p['actual_readiness']['path']);assert sha(R)==p['actual_readiness']['sha256'];ready=json.loads(R.read_text());assert ready['HF']==new['model_path']
 for item in ready['metadata'].values():assert Path(item['path']).stat().st_size==item['bytes']and sha(Path(item['path']))==item['sha256']
 guard={x['path']:x['sha256']for x in new['model_guard']}
 keys=['merge-admission.json','merge-manifest.json','model.safetensors.index.json','config.json']if label=='M51'else['event.json','audit.json','index.json','config.json']
 for key in keys:
  item=ready['metadata'][key];assert guard[item['uri']]==item['sha256']
 archive=Path(p['source_archive']['path']).read_bytes();assert hashlib.sha256(archive).hexdigest()==p['source_archive']['sha256'];rootreq=math_only.submission(name=t['root_name'],candidate=new,archive=archive,credentials=dict(CW_KEY_ID='fixture',CW_KEY_SECRET='fixture'));inspect.signature(IrisClient.submit).bind(None,**rootreq);assert rootreq['entrypoint'].workdir_files=={math_only.ARCHIVE_PATH:archive};assert rootreq['timeout'].to_seconds()==49*3600 and rootreq['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION and rootreq['max_retries_failure']==rootreq['max_retries_preemption']==0
 captured=[];fakejob=SimpleNamespace(job_id=JobName.from_wire('/fixture/owned-serve'));fakeclient=object.__new__(IrisClient);fakeclient._cluster_client=SimpleNamespace(shutdown=lambda **_:None);fakeclient._controller=None;native_signature=inspect.signature(IrisClient.submit)
 def capture(client,*args,**kw):
  native_signature.bind(client,*args,**kw);assert kw['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION and kw['max_retries_failure']==kw['max_retries_preemption']==0 and kw['timeout'].to_seconds()==43200;captured.append(kw);return fakejob
 info=SimpleNamespace(controller_address='fixture-only',credentials=None,bundle_id='fixture-bundle')
 with patch.object(IrisClient,'submit',capture),patch.object(IrisClient,'in_cluster',return_value=fakeclient),patch.object(owner,'iris_ctx',return_value=SimpleNamespace(client=SimpleNamespace(job=lambda _:fakejob))):
  assert owner.submit_serve(new,info)is fakejob
 assert len(captured)==1
 native=create_workspace_zip(W);views=[]
 with tempfile.TemporaryDirectory(prefix=label+'-math-native-')as td,zipfile.ZipFile(io.BytesIO(native))as z:
  dest=Path(td);z.extractall(dest);assert z.read(t['committed_path'])==(W/t['committed_path']).read_bytes();env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONSAFEPATH='1',IRIS_PYTHON=sys.executable,IRIS_VENV=sys.prefix,IRIS_WORKDIR=str(dest));env['PYTHONPATH']=':'.join([str(dest),*[str(q)for q in (dest/'lib').glob('*/src')]])
  command=rootreq['entrypoint'].command.copy();command[-1]+=' --inspect-only';ran=subprocess.run(command,cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90);assert ran.stdout.strip().splitlines()[-1]==new['label']
  for trial in ['greedy','17','29']:
   with patch.dict(os.environ,IRIS_WORKDIR=str(dest)):entry=owner.task_entrypoint(new,'math',trial)
   command=entry.command.copy();command[-1]+=' --inspect-only';ran=subprocess.run(command,cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90);view=json.loads(ran.stdout.strip().splitlines()[-1]);assert view['candidate_sha256']==t['candidate_sha256']and view['command_argv']==next(x['argv']for x in new['math']if x['trial']==trial);views.append(view)
 out=dict(result='PASS',code_pin=p['code_pin'],native_bundle_sha256=hashlib.sha256(native).hexdigest(),native_bundle_bytes=len(native),priority_only_owner_function_AST_delta=True,scientific_driver_signature_native_f5_and_frozenserve_bytes_exact=True,candidate_command_deltas_only_model_namespaces_priority=True,actual_readiness_and_metadata_guard_joins=True,actual_production_root_and_ownedserve_constructor=True,root_command=rootreq['entrypoint'].command,ownedserve_command=captured[0]['entrypoint'].command,all3_actual_extracted_children=views,new_reference_serve=False,actual_runtime_sampling_or_HF_attestation=False,remote_model_storage_or_submit_calls=0)
 (H/'native-binding-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['all3_actual_extracted_children','ownedserve_command','root_command']}))
if __name__=='__main__':main()
