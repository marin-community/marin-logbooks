"""Resolve all real native root/train/export entrypoints and unchanged BFCL configs."""
from pathlib import Path
import copy,hashlib,inspect,json,io,subprocess,zipfile
from iris.cluster.client.bundle import create_workspace_zip
import yaml
from iris.client.client import IrisClient
from experiments.post_training.identity_path_b.ablation_launch import submission,workflow_worker
from experiments.post_training.identity_path_b.ablation import train_worker,export_worker,AblationArm,check_report,frozen_names,freezes_special_rows
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 plan=json.loads((HERE/'launch-plan.json').read_text());runtime=Path(plan['runtime']['local']).read_bytes();assert hashlib.sha256(runtime).hexdigest()==plan['runtime']['sha256']
 stage=json.loads((HERE/'stage-files.json').read_text());source=json.loads((HERE.parent/'full-screen-execution-r1/execution-packet.json').read_text());template=next(x for x in source['candidates'] if x['label']=='pathb-base25-u52-r1')['bfcl']
 proof=[]
 for arm in plan['arms']:
  training=copy.deepcopy(arm['training_kwargs']);training['runtime_bytes']=runtime
  exposure_item=next(i for i in stage['files'] if i['uri']==training['exposure_path']);report=json.loads(Path(exposure_item['local']).read_text());check_report(report,AblationArm(arm['arm']))
  assert sha(Path(exposure_item['local']))==training['exposure_sha256']
  request=submission(name=arm['root_name'],training=training,credentials={'CW_KEY_ID':'fixture-only','CW_KEY_SECRET':'fixture-only'})
  inspect.signature(IrisClient.submit).bind(None,**request)
  fn,args,kwargs=request['entrypoint'].resolve();assert fn is workflow_worker and kwargs=={'training':training};inspect.signature(fn).bind(*args,**kwargs)
  inspect.signature(train_worker).bind(**training)
  payload=arm['final_checkpoint'];event=CheckpointEvent(**payload);event.payload()
  assert event.exposure==report['cumulative_batches'][-1] and event.completed_updates==36 and event.native_checkpoint.endswith('/step-35')
  export={k:training[k] for k in ['run_id','runtime_bytes','runtime_sha256','source_receipt','source_objects']}
  inspect.signature(export_worker).bind(event=event,**export)
  assert len(frozen_names(AblationArm(arm['arm'])))==arm['frozen_native_tensors']
  assert freezes_special_rows(AblationArm(arm['arm']))==arm['special_rows_frozen']
  assert request['timeout'].to_seconds()==176400 and request['max_retries_failure']==request['max_retries_preemption']==0
  method=arm['bfcl'];old=yaml.safe_load(Path(template['model_config']).read_text());new=yaml.safe_load(Path(method['model_config']).read_text())
  assert {k:v for k,v in old.items() if k not in ['name','location']}=={k:v for k,v in new.items() if k not in ['name','location']}
  assert new['location']==event.hf_export
  a=list(method['argv']);b=list(template['argv'])
  for flag in ['--model-config','--version','--records-prefix','--description']:
   a[a.index(flag)+1]=b[b.index(flag)+1]
  assert a==b
  assert sha(Path(method['policy_wrapper']))==method['policy_wrapper_sha256']==template['policy_wrapper_sha256']
  proof.append({'arm':arm['arm'],'root_name':arm['root_name'],'native_root_train_export_kwargs':'PASS','actual_final36_native35':'PASS','model_yaml_and_science_argv_identity_only_delta':'PASS','frozen_native':arm['frozen_native_tensors'],'trained_master_cast':arm['trained_master_cast_tensors']})
 root=Path(plan['workspace']);old_pin='6ac2c53d061156ad22be53b332b7bd677a856290'
 changed=subprocess.check_output(['git','diff','--name-only',old_pin,plan['code_pin']],cwd=root,text=True).splitlines();assert changed==['experiments/post_training/identity_path_b/ablation.py']
 name=changed[0];old_source=subprocess.check_output(['git','show',old_pin+':'+name],cwd=root).decode();assert (root/name).read_text()==old_source.replace('    T3_BRIDGE = \"A4\"','    T3_BRIDGE = \"A4\"\n    GLM_ON_OFF_FALLBACK = \"A6\"')
 bundle=create_workspace_zip(root)
 with zipfile.ZipFile(io.BytesIO(bundle)) as archive:assert archive.read(name)==(root/name).read_bytes();members=len(archive.namelist())
 out={'native_bundle_sha256':hashlib.sha256(bundle).hexdigest(),'native_bundle_bytes':len(bundle),'members':members,'only_source_delta':'one enum value','plan_sha256':sha(HERE/'launch-plan.json'),'code_pin':plan['code_pin'],'arms':proof,'live_calls':0}
 (HERE/'native-A6-bindings-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
