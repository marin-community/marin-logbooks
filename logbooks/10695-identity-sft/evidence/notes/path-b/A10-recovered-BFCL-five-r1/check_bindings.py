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
 ready=json.loads(Path(plan['actual_readiness']['path']).read_bytes());assert sha(Path(plan['actual_readiness']['path']))==plan['actual_readiness']['sha256'];target=plan['targets']['A10'];assert ready['HF']==target['location']
 for x in ready['metadata'].values():assert Path(x['path']).stat().st_size==x['bytes'] and sha(Path(x['path']))==x['sha256']
 event=json.loads(Path(ready['metadata']['event.json']['path']).read_bytes());assert all(event[k]==v for k,v in target['event'].items());assert ready['metadata']['event.json']['uri']==target['location']+'/path-b-checkpoint-event.json'
 audit=json.loads(Path(ready['metadata']['audit.json']['path']).read_bytes());assert len(audit['tensors'])==502 and audit['frozen_native_tensors']==104 and audit['trained_master_cast_tensors']==398
 orig=json.loads((HERE.parent/'ablation-a10-r1/launch-plan.json').read_bytes())['arms'][0]['bfcl'];a=yaml.safe_load(Path(orig['model_config']).read_bytes());b=yaml.safe_load(Path(target['model_config']).read_bytes());assert {k:v for k,v in a.items() if k not in ['name','location']}=={k:v for k,v in b.items() if k not in ['name','location']}
 spec=importlib.util.spec_from_file_location('a10_freshbfcl',HERE/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
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
  for run in range(1,6):
   batch=m.prepare('A10',run);expected=target['runs'][run-1];assert batch.model.location==target['location'] and batch.records_prefix==expected['records_prefix'] and os.environ['MARIN_PREFIX']==expected['MARIN_PREFIX'];assert len(batch.evaluations)==1
   e=batch.evaluations[0];assert e.identity.output_dir.startswith(expected['records_prefix']+'/') and e.identity.output_dir not in dirs;dirs.add(e.identity.output_dir)
   c=Capture();m.submit(batch,c);assert c.batch==batch
   out.append(dict(run=run,records_prefix=batch.records_prefix,output_dir=e.identity.output_dir,model=batch.model.location,task_config=asdict(e.executor)))
 finally:held_bfcl.resolve_secret_spec=oldresolver;held_bfcl.env_vars_from_keys=oldenv;launch.preflight_harbor_configs=original_preflight
 proof=dict(actual_readiness_sha256=plan['actual_readiness']['sha256'],actual502_and6metadata_hashjoins='PASS',native5requests_and_SDKnamespace='PASS',all5=out,actual_first_Harbor_preflight=True,remaining4_proof_only_cached_exact_same_config=True,source_science='UNCHANGED CABA config bytes/YAMLonlyname+location',remote_model_storage_calls=0)
 (HERE/'native-five-bindings-proof.json').write_text(json.dumps(proof,indent=2,default=str)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='all5'}))
if __name__=='__main__':main()
