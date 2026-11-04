from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace
import ast,hashlib,inspect,io,json,os,subprocess,sys,tempfile,zipfile
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2
from experiments.post_training.identity_path_b import a12_export_recovery as recovery
from experiments.post_training.identity_path_b import async_launch
from experiments.post_training.identity_path_b.ablation import export_worker
from experiments.post_training.identity_path_b.workflow import EXPORT_RESOURCES,EXPORT_LIFETIME
H=Path(__file__).resolve().parent;B=H.parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
 plan=json.loads((H/'launch-plan.json').read_text());W=Path(plan['workspace']);pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip();assert pin==plan['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
 changed=subprocess.check_output(['git','diff','--name-only',plan['parent_pin'],pin],cwd=W,text=True).splitlines();assert changed==['experiments/post_training/identity_path_b/a12_export_binding.json','experiments/post_training/identity_path_b/a12_export_recovery.py']
 for name in ['ablation.py','ablation_launch.py','async_launch.py','checkpoints.py','export_preflight.py','worker.py','storage_audit.py','workflow.py','rehydrate.py']:
  path='experiments/post_training/identity_path_b/'+name;assert subprocess.check_output(['git','show',plan['parent_pin']+':'+path],cwd=W)==(W/path).read_bytes()
 item=recovery.binding('A12');original=recovery.checkpoint_event(item['original_event']);event=recovery.checkpoint_event(item['event']);runtime=Path(plan['runtime']['local']).read_bytes();assert sha(runtime)==plan['runtime']['sha256']
 old=item['original_training'];new=item['training'];assert {k for k in old if old[k]!=new[k]}=={'run_root','run_id'};assert event==replace(original,hf_export=item['root']+'/hf/updates-18',eval_scratch=item['root']+'/eval/updates-18')
 localmap=json.loads((H/'local-evidence-map.json').read_text());stage=json.loads((B/'ablation-a12-a13-r1/stage-files.json').read_text())['files'];localmap.update({x['uri']:x['local'] for x in stage})
 def pinned(uri,digest):
  raw=Path(localmap[uri]).read_bytes();assert sha(raw)==digest;return raw
 with patch.object(recovery,'read_pinned',pinned):
  actual,e=recovery.checked_original_training();assert e==original and actual['training_result']['completed_updates']==18
  for bad in [replace(event,native_checkpoint=event.native_checkpoint.replace('step-17','step-16')),original,replace(event,hf_export=event.hf_export.replace('/bfcl-ablation-','/unsafe-'))]:
   with patch.object(recovery,'checked_export_event',side_effect=AssertionError('Invalid retirement reached IO')):
    try:recovery.retire_recovered_native(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid retirement accepted')
 req=recovery.recovery_submission(runtime_bytes=runtime,credentials=dict(CW_KEY_ID='fixture-only',CW_KEY_SECRET='fixture-only'));inspect.signature(IrisClient.submit).bind(None,**req)
 fn,args,kwargs=req['entrypoint'].resolve();assert fn is recovery.export_only_worker and kwargs==dict(runtime_bytes=runtime);assert req['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION and req['max_retries_failure']==req['max_retries_preemption']==0 and req['timeout'].to_seconds()==176400 and req['constraints'] and req['entrypoint'].workdir_files
 childentry=Entrypoint.from_callable(export_worker,event=event,**{k:{**new,'runtime_bytes':runtime}[k] for k in ['run_id','runtime_bytes','runtime_sha256','source_receipt','source_objects']});cf,ca,ck=childentry.resolve();inspect.signature(cf).bind(*ca,**ck);assert cf is export_worker
 captured=[]
 def submit(**kw):inspect.signature(IrisClient.submit).bind(None,**kw);captured.append(kw);return 'native-handle-fixture'
 with patch.object(recovery,'regional_task_context',return_value=SimpleNamespace(client=SimpleNamespace(submit=submit))):
  assert recovery.native_submit(childentry,name='export-18',resources=EXPORT_RESOURCES,replicas=1,env_vars={'JAX_PLATFORMS':'cuda,cpu'},lifetime=EXPORT_LIFETIME)=='native-handle-fixture'
 child=captured[0];assert child['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION and child['max_retries_failure']==child['max_retries_preemption']==0 and child['timeout'].to_seconds()==7200 and child['replicas']==1
 # Same implementation as native_submit except authorized priority and annotations/docstring.
 def fn_ast(f):
  node=ast.parse(inspect.getsource(f)).body[0];node.body=node.body[1:] if isinstance(node.body[0],ast.Expr) else node.body
  node.args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]);node.returns=None
  for n in ast.walk(node):
   if isinstance(n,ast.Attribute) and n.attr=='PRIORITY_BAND_INTERACTIVE':n.attr='PRIORITY_BAND_PRODUCTION'
  return ast.dump(node,include_attributes=False)
 assert fn_ast(recovery.native_submit)==fn_ast(async_launch.native_submit)
 bundle=create_workspace_zip(W);(H/'author-sealed-workspace.zip').write_bytes(bundle)
 with zipfile.ZipFile(io.BytesIO(bundle)) as z,tempfile.TemporaryDirectory(prefix='a12-export-native-') as td:
  dest=Path(td);z.extractall(dest)
  for name in changed:assert z.read(name)==(W/name).read_bytes()
  env=dict(os.environ,PYTHONSAFEPATH='1',PYTHONPATH=':'.join([str(dest),*[str(p) for p in (dest/'lib').glob('*/src')]]));code='from experiments.post_training.identity_path_b.a12_export_recovery import binding; print(binding("A12")["job_name"])'
  result=subprocess.run([sys.executable,'-P','-c',code],cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90);assert result.stdout.strip()==item['job_name'];members=len(z.namelist())
 out=dict(result='PASS',code_pin=pin,parent_scientific_modules_exact=True,original18_native17_finite_dispatch_evidence=True,invalid_retirement_before_IO=True,priority_only_native_submit_AST_parity=True,actual_root=dict(name=req['name'],priority='production',command=req['entrypoint'].command,callable_files=list(req['entrypoint'].workdir_files),timeout_seconds=176400,retries=[0,0]),actual_export_child=dict(name=child['name'],priority='production',command=childentry.command,callable_files=list(childentry.workdir_files),timeout_seconds=7200,retries=[0,0]),native_bundle_sha256=sha(bundle),native_bundle_bytes=len(bundle),members=members,extracted_committed_binding_import=True,remote_calls=0,actual_recovered_HF_readiness_claimed=False)
 (H/'native-binding-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
