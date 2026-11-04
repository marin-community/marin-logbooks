from pathlib import Path
import ast,hashlib,inspect,io,json,os,subprocess,sys,tempfile,zipfile,yaml
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from experiments.post_training.identity_path_b import write_recovery as recovery
from experiments.post_training.identity_path_b.ablation_launch import workflow_worker
from experiments.post_training.identity_path_b.ablation import train_worker,export_worker,check_report,AblationArm
HERE=Path(__file__).resolve().parent;B=HERE.parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
 plan=json.loads((HERE/'launch-plan.json').read_text());root=Path(plan['workspace']);pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert pin==plan['code_pin'] and not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
 changed=subprocess.check_output(['git','diff','--name-only',plan['original_scientific_pin'],pin],cwd=root,text=True).splitlines();assert changed==['experiments/post_training/identity_path_b/write_recovery.py','experiments/post_training/identity_path_b/write_recovery_binding.json']
 for file in ['ablation.py','ablation_launch.py','async_launch.py','checkpoints.py','export_preflight.py','worker.py','storage_audit.py','workflow.py']:
  path='experiments/post_training/identity_path_b/'+file;assert subprocess.check_output(['git','show',plan['original_scientific_pin']+':'+path],cwd=root)==(root/path).read_bytes()
 original_a9=subprocess.check_output(['git','show','8c44a21a1aae7afb67277b1b048f92b57f88cb20:experiments/post_training/identity_path_b/ablation.py'],cwd=root,text=True)
 def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 assert functions(original_a9)==functions((root/'experiments/post_training/identity_path_b/ablation.py').read_text())
 runtime=Path(plan['runtime']['local']).read_bytes();assert sha(runtime)==plan['runtime']['sha256'];creds={'CW_KEY_ID':'fixture-only','CW_KEY_SECRET':'fixture-only'};proof=[]
 for label,item in plan['arms'].items():
  old=item['original_training'];new=item['training'];assert {k:v for k,v in old.items() if k not in ['run_id','run_root']}=={k:v for k,v in new.items() if k not in ['run_id','run_root']};assert new['run_root']!=old['run_root']
  report_item=next(x for x in json.loads((B/f'ablation-{label.lower()}-r1/stage-files.json').read_text())['files'] if x['uri']==new['exposure_path']);report_raw=Path(report_item['local']).read_bytes();assert sha(report_raw)==new['exposure_sha256'];check_report(json.loads(report_raw),AblationArm(label))
  req=recovery.recovery_submission(label=label,runtime_bytes=runtime,credentials=creds);inspect.signature(IrisClient.submit).bind(None,**req);fn,args,kwargs=req['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kwargs)
  assert req['name']==item['job_name'] and req['timeout'].to_seconds()==176400 and req['max_retries_failure']==req['max_retries_preemption']==0 and req['constraints'] and req['entrypoint'].workdir_files
  if label=='A9':assert fn is recovery.export_only_worker and kwargs==dict(runtime_bytes=runtime)
  else:assert fn is workflow_worker and kwargs==dict(training={**new,'runtime_bytes':runtime});inspect.signature(train_worker).bind(**kwargs['training'])
  event=recovery.checkpoint_event(item['event']);original=recovery.checkpoint_event(item['original_event']);assert event.exposure==original.exposure and event.training_pin==original.training_pin and event.packed_data_sha256==original.packed_data_sha256
  assert event.hf_export!=original.hf_export and event.eval_scratch!=original.eval_scratch
  if label=='A9':assert event.native_checkpoint==original.native_checkpoint
  else:assert event.native_checkpoint!=original.native_checkpoint
  inspect.signature(export_worker).bind(event=event,**{k:{**new,'runtime_bytes':runtime}[k] for k in ['run_id','runtime_bytes','runtime_sha256','source_receipt','source_objects']})
  oldbf=json.loads((B/f'ablation-{label.lower()}-r1/launch-plan.json').read_text())['arms'][0]['bfcl'];newbf=item['BFCL'];oldyaml=yaml.safe_load(Path(oldbf['model_config']).read_text());newyaml=yaml.safe_load(Path(newbf['model_config']).read_text());assert {k:v for k,v in oldyaml.items() if k not in ['name','location']}=={k:v for k,v in newyaml.items() if k not in ['name','location']};assert newyaml['location']==event.hf_export
  a=list(newbf['argv']);b=list(oldbf['argv'])
  for flag in ['--model-config','--version','--records-prefix','--description']:a[a.index(flag)+1]=b[b.index(flag)+1]
  assert a==b and newbf['policy_wrapper_sha256']==oldbf['policy_wrapper_sha256']
  proof.append(dict(arm=label,root_name=req['name'],new_HF=event.hf_export,native=event.native_checkpoint,actual_callable_kwargs='PASS',original_science_data_runtime_bytes='PASS',BFCL_model_and_execution_identity_only_deltas='PASS',native_callable_workdir_files_preserved=True))
 bundle=create_workspace_zip(root)
 with zipfile.ZipFile(io.BytesIO(bundle)) as archive,tempfile.TemporaryDirectory(prefix='write-recovery-native-ZIP-') as td:
  dest=Path(td);archive.extractall(dest)
  for name in changed:assert archive.read(name)==(root/name).read_bytes()
  env=dict(os.environ,PYTHONSAFEPATH='1',JAX_PLATFORMS='cpu');env['PYTHONPATH']=':'.join([str(dest),*[str(p) for p in (dest/'lib').glob('*/src')]])
  code='import json; from experiments.post_training.identity_path_b.write_recovery import binding; print(json.dumps({x:binding(x)["job_name"] for x in ["A9","A10"]}))'
  ran=subprocess.run([sys.executable,'-P','-c',code],cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90);assert json.loads(ran.stdout)=={label:item['job_name'] for label,item in plan['arms'].items()}
  members=len(archive.namelist())
 out=dict(result='PASS',source_pin=pin,native_bundle_sha256=sha(bundle),native_bundle_bytes=len(bundle),members=members,arms=proof,actual_archive_module_and_committed_binding_import='PASS',original_A9_all_functions_equal=True,original_A10_scientific_modules_exact_bytes=True,live_calls=0,actual_new_HF_readiness_claimed=False)
 (HERE/'native-recovery-bindings-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
