from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import ast,hashlib,importlib.util,inspect,io,json,os,subprocess,zipfile,yaml
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import job_pb2
from experiments.evaluation import held_bfcl,launch
from marin.evaluation.harbor.driver_config import ValidatedHarborConfig,HarborDatasetKind,HarborErrorTaxonomy
from marin.evaluation.records import BenchmarkMetadataRef,BenchmarkMetricRef,MetricKind
H=Path(__file__).resolve().parent;B=H.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((H/'launch-plan.json').read_text());W=Path(p['workspace']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==p['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
 label=next(iter(p['targets']));t=p['targets'][label];ready_path=Path(p['actual_readiness']['path']);assert sha(ready_path)==p['actual_readiness']['sha256'];ready=json.loads(ready_path.read_text());assert ready['HF']==t['location']
 for x in ready['metadata'].values():assert Path(x['path']).stat().st_size==x['bytes']and sha(Path(x['path']))==x['sha256']
 assert ready['actual_RNO_bucket_landing_verified_NOT_just_declared_URI']is True
 index=json.loads(Path(ready['metadata']['index.json']['path']).read_text());heads=ready['indexed39_shards_actual_RNO_object_HEADs'];assert len(index['weight_map'])==502 and len(set(index['weight_map'].values()))==len(heads)==39;assert {x['uri'] for x in heads}=={t['location']+'/'+n for n in index['weight_map'].values()};assert all(x['bytes']>0 and x['actual_object_name'].startswith('marin-rno2a/')for x in heads)
 audit=json.loads(Path(ready['metadata']['audit.json']['path']).read_text());assert len(audit['tensors'])==502 and audit['frozen_native_tensors']==104 and audit['trained_master_cast_tensors']==398
 event=json.loads(Path(ready['metadata']['event.json']['path']).read_text());assert event==t['actual_event'] and event['arm']=='A12'and event['completed_updates']==18 and event['hf_export']==t['location']and event['native_checkpoint'].endswith('/bfcl-ablation-69ddc88fb2ff/A12/run/checkpoints/step-17')
 orig=json.loads((B/'ablation-a12-a13-r1/launch-plan.json').read_text())['arms'][0]['bfcl'];a=yaml.safe_load(Path(orig['model_config']).read_text());b=yaml.safe_load(Path(t['model_config']).read_text());assert {k:v for k,v in a.items()if k not in ['name','location']}=={k:v for k,v in b.items()if k not in ['name','location']}

 admission=p['qualified_replacement'];ep=Path(admission['evidence']['path']);assert sha(ep)==admission['evidence']['sha256'];ev=json.loads(ep.read_text());assert ev['SDK_terminal']is True and ev['logical_full123_admissible']is False and ev['source_HF']==t['location']and ev['original_slot']==1 and ev['replacement_number']==0
 ref=p['original_slot1_cohort'];assert sha(Path(ref['path']))==ref['sha256'];cohort=json.loads(Path(ref['path']).read_text());assert cohort['actual_original_job_id_expected_and_guarded']==ev['job']and cohort['frozen_before_submit_AND_any_transcript_read']is True;ids=set(cohort['all123_doc_ids']);assert len(ids)==123 and {x['doc_id']for x in ev['rows']}==ids and sum(x['reward']is not None for x in ev['rows'])==122
 sample=ev['samples_full_JSON_native_reader_projection'];assert sha(Path(sample['path']))==sample['sha256'];samples=json.loads(Path(sample['path']).read_text());assert len(samples)==123 and {x['doc_id']for x in samples}==ids
 missing=[]
 for row in samples:
  detail=json.loads(row['grading']['detail'])
  if detail['scored']is False:missing.append((row['doc_id'],detail['error']['type'],detail['error']['category']));assert row['grading']['score']is None and row['metrics']==[]
  else:assert row['grading']['score']in [0.,1.]and detail['rewards']['reward']==row['grading']['score']
 assert missing==[('bfcl-simple-python-161','EnvironmentStartTimeoutError','infrastructure')]
 original=t['runs'][0]['original_configuration'];fresh=t['runs'][0];assert fresh['replacement_number']==1 and fresh['run']==original['run']==1 and fresh['records_prefix']!=original['records_prefix']and fresh['MARIN_PREFIX']!=original['MARIN_PREFIX']and fresh['version']!=original['version']
 spec=importlib.util.spec_from_file_location('packet',H/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 original_proof=B/'A11-BFCL-five-r1/native-five-bindings-proof.json';cached=json.loads(original_proof.read_text())['all5'][0]['task_config']['config'];tax=cached['error_taxonomy'];cached['error_taxonomy']=HarborErrorTaxonomy(**{k:(v if k=='commit'else frozenset(ast.literal_eval(v[len('frozenset('):-1])))for k,v in tax.items()});bm=BenchmarkMetadataRef(schema_version=1,task='bfcl_parity',primary_metric='reward',metric_kind=MetricKind.CONTINUOUS,metrics=(BenchmarkMetricRef(name='reward',source_name='reward',kind=MetricKind.CONTINUOUS,higher_is_better=True),),n_benchmark=123,n_attempted=123);assert str(bm)==cached['benchmark'];cached.update(benchmark=bm,dataset_kind=HarborDatasetKind(cached['dataset_kind']),verifier_env_keys=tuple(cached['verifier_env_keys']));fixture=ValidatedHarborConfig(**cached)
 oldpre=launch.preflight_harbor_configs;oldresolve=held_bfcl.resolve_secret_spec;oldenv=held_bfcl.env_vars_from_keys
 def preflight(requests,**kw):assert all(Path(x).read_bytes()==Path(p['suite']['path']).read_bytes()for x,_ in requests);return [fixture for _ in requests]
 launch.preflight_harbor_configs=preflight;held_bfcl.resolve_secret_spec=lambda _:SimpleNamespace(value='fixture-only');held_bfcl.env_vars_from_keys=lambda _:dict(MARIN_PREFIX=os.environ['MARIN_PREFIX']);proofs=[];dirs=set()
 class Capture:
  def submit(self,**kw):
   inspect.signature(IrisClient.submit).bind(None,**kw);assert kw['timeout'].to_seconds()==176400 and kw['max_retries_failure']==kw['max_retries_preemption']==0 and kw['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION;assert kw['environment'].env_vars['MARIN_PREFIX']==os.environ['MARIN_PREFIX'];fn,args,k=kw['entrypoint'].resolve();assert fn is held_bfcl.run_batch and not k;self.batch=args[0];return SimpleNamespace(job_id='fixture-only')
 try:
  for n in [1]:
   batch=m.prepare(label,n);run=t['runs'][n-1];assert batch.priority_band==job_pb2.PRIORITY_BAND_PRODUCTION and batch.records_prefix==run['records_prefix']and batch.model.location==t['location'];e=batch.evaluations[0];assert e.identity.output_dir.startswith(run['records_prefix']+'/')and e.identity.output_dir not in dirs;dirs.add(e.identity.output_dir);assert json.loads(e.executor.config.stable_policy_json)==json.loads(fixture.stable_policy_json)
   serve=held_bfcl.inference_config_for_model(batch.model,batch.accelerator,env_vars={},priority=batch.priority_band,capability_origin=batch.capability_origin,api_model=batch.api_model);assert serve.iris.priority==job_pb2.PRIORITY_BAND_PRODUCTION;c=Capture();m.submit(batch,c);assert c.batch==batch;proofs.append(dict(run=n,records_prefix=run['records_prefix'],output_dir=e.identity.output_dir,task_config=asdict(e.executor),priority_root_and_serve='production'))
 finally:launch.preflight_harbor_configs=oldpre;held_bfcl.resolve_secret_spec=oldresolve;held_bfcl.env_vars_from_keys=oldenv
 z=create_workspace_zip(W);assert hashlib.sha256(z).hexdigest()=='5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b'
 out=dict(whole_run_original_slot1_replacement1=True,original_first_slot1_diagnostic_unchanged=True,result='PASS',code_pin=p['code_pin'],native_bundle_sha256=hashlib.sha256(z).hexdigest(),native_bundle_bytes=len(z),actual_readiness_sha256=p['actual_readiness']['sha256'],actual_metadata_and_RNO_landing_502_index_joins=True,unchanged_model_profile_except_name_location=True,all5=proofs,preflight_scope='Replay exact sealed prior validated fixed123 config at proof boundary; actual prepare still performs standard native preflight.',remote_calls=0)
 (H/'native-one-replacement-binding-proof.json').write_text(json.dumps(out,indent=2,default=str)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='all5'}))
if __name__=='__main__':main()
