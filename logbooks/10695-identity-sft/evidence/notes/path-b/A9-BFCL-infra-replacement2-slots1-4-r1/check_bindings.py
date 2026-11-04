from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import hashlib,importlib.util,inspect,json,os,subprocess
import yaml
from iris.client.client import IrisClient
from experiments.evaluation import held_bfcl,launch
HERE=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 plan=json.loads((HERE/'launch-plan.json').read_bytes());root=Path(plan['workspace']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin']
 ready=json.loads(Path(plan['actual_readiness']['path']).read_bytes());assert sha(Path(plan['actual_readiness']['path']))==plan['actual_readiness']['sha256'];target=plan['targets']['A9'];assert ready['HF']==target['location']
 for x in ready['metadata'].values():assert Path(x['path']).stat().st_size==x['bytes'] and sha(Path(x['path']))==x['sha256']
 event=json.loads(Path(ready['metadata']['event.json']['path']).read_bytes());assert all(event[k]==v for k,v in target['event'].items());assert ready['metadata']['event.json']['uri']==target['location']+'/path-b-checkpoint-event.json'
 audit=json.loads(Path(ready['metadata']['audit.json']['path']).read_bytes());assert len(audit['tensors'])==502 and audit['frozen_native_tensors']==104 and audit['trained_master_cast_tensors']==398
 orig=json.loads((HERE.parent/'ablation-a9-r1/launch-plan.json').read_bytes())['arms'][0]['bfcl'];a=yaml.safe_load(Path(orig['model_config']).read_bytes());b=yaml.safe_load(Path(target['model_config']).read_bytes());assert {k:v for k,v in a.items() if k not in ['name','location']}=={k:v for k,v in b.items() if k not in ['name','location']}
 assert [item['run'] for item in target['runs']]==[1, 4] and plan['policy']['no_R3']
 index=Path(plan['replacement_admission_index']['path']);assert sha(index)==plan['replacement_admission_index']['sha256'];rows=json.loads(index.read_bytes())['physical_runs']
 assert sha(Path(plan['parent_original_packet']['path']))==plan['parent_original_packet']['sha256']
 cohort=Path(plan['original_slot1_cohort']['path']);assert sha(cohort)==plan['original_slot1_cohort']['sha256'];sourceIDs=set(json.loads(cohort.read_bytes())['doc_ids']);assert len(sourceIDs)==123
 for item in target['runs']:
  assert item['replacement_number']==2 and item['no_R3']
  old=item['original_configuration'];r1=item['previous_R1_configuration'];assert item['version']==old['version']+'-infra-replacement2'
  assert sha(Path(item['previous_R1_packet']['path']))==item['previous_R1_packet']['sha256']
  for key in ['records_prefix','MARIN_PREFIX']:
   assert item[key]==old[key].replace(f"/run-{item['run']}/",f"/run-{item['run']}-infra-replacement-2/")
   assert len({item[key],old[key],r1[key]})==3
  for number,key in [(0,'original_admission'),(1,'previous_R1_admission')]:
   row=item[key];assert row in rows and row['replacement_number']==number and row['state']=='TERMINAL_INCOMPLETE_NATIVE123_NOT_PRIMARY_SCORE' and row['score'] is None and row['scored_binary_rewards']<123
   launchpath=Path(row['launch']);assert sha(launchpath)==row['launch_sha256'];array=Path(row['partial_array']);assert sha(array)==row['partial_array_sha256']
   arrayrows=json.loads(array.read_bytes())['rows'];assert len(arrayrows)==123 and {r['doc_id'] for r in arrayrows}==sourceIDs
   finite=[r['reward'] for r in arrayrows if r['score_present']];missing=[r for r in arrayrows if not r['score_present']]
   assert all(v in [0,1] for v in finite) and len(finite)==row['scored_binary_rewards'] and sum(finite)==row['observed_correct'] and all(r['reward'] is None for r in missing)
  prior=json.loads(Path(item['previous_R1_admission']['launch']).read_bytes());assert prior['packet_sha256']==item['previous_R1_packet']['sha256']
  assert prior['records_prefix']==r1['records_prefix'] and prior['MARIN_PREFIX']==r1['MARIN_PREFIX']
 spec=importlib.util.spec_from_file_location('a9_freshbfcl',HERE/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 original_preflight=launch.preflight_harbor_configs;cached=[]
 def cached_preflight(requests,**kwargs):
  assert all(Path(path).read_bytes()==Path(plan['suite']['path']).read_bytes() for path,_ in requests)
  if not cached:cached.extend(original_preflight(requests,**kwargs))
  return cached
 launch.preflight_harbor_configs=cached_preflight
 oldresolver=held_bfcl.resolve_secret_spec;oldenv=held_bfcl.env_vars_from_keys;held_bfcl.resolve_secret_spec=lambda spec:SimpleNamespace(value='fixture-only');held_bfcl.env_vars_from_keys=lambda keys:{'MARIN_PREFIX':os.environ['MARIN_PREFIX']}
 class Capture:
  def submit(self,**kwargs):
   inspect.signature(IrisClient.submit).bind(None,**kwargs);assert kwargs['timeout'].to_seconds()==176400 and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;assert kwargs['environment'].env_vars['MARIN_PREFIX']==os.environ['MARIN_PREFIX'];fn,args,kw=kwargs['entrypoint'].resolve();assert fn is held_bfcl.run_batch and not kw;self.batch=args[0];return SimpleNamespace(job_id='fixture')
 out=[];dirs=set()
 try:
  for expected in target['runs']:
   run=expected['run']
   batch=m.prepare('A9',run);assert batch.model.location==target['location'] and batch.records_prefix==expected['records_prefix'] and os.environ['MARIN_PREFIX']==expected['MARIN_PREFIX'];assert len(batch.evaluations)==1
   e=batch.evaluations[0];assert e.identity.output_dir.startswith(expected['records_prefix']+'/') and e.identity.output_dir not in dirs;dirs.add(e.identity.output_dir)
   c=Capture();m.submit(batch,c);assert c.batch==batch
   out.append(dict(run=run,records_prefix=batch.records_prefix,output_dir=e.identity.output_dir,model=batch.model.location,task_config=asdict(e.executor)))
 finally:held_bfcl.resolve_secret_spec=oldresolver;held_bfcl.env_vars_from_keys=oldenv;launch.preflight_harbor_configs=original_preflight
 proof=dict(actual_readiness_sha256=plan['actual_readiness']['sha256'],actual502_and6metadata_hashjoins='PASS',native2requests_and_SDKnamespace='PASS',selected_requests=out,actual_first_Harbor_preflight=True,remaining1_proof_only_cached_exact_same_config=True,source_science='UNCHANGED CABA config bytes/YAMLonlyname+location',remote_model_storage_calls=0)
 (HERE/'native-final-R2-bindings-proof.json').write_text(json.dumps(proof,indent=2,default=str)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='selected_requests'}))
if __name__=='__main__':main()
