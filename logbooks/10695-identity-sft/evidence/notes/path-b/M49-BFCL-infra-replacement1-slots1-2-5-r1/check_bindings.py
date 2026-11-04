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
 assert ready['provenance_kind']=='linear-cpu-merge' and ready['fake_training_event_or_master_cast_audit']is False
 report=json.loads(Path(ready['metadata']['merge-admission.json']['path']).read_text());manifest=json.loads(Path(ready['metadata']['merge-manifest.json']['path']).read_text());index=json.loads(Path(ready['metadata']['model.safetensors.index.json']['path']).read_text());assert report['admitted']is True and report['label']==label and report['output']==t['location']and report['training_event']is None and report['master_cast_audit']is None
 assert len(report['tensors'])==len(index['weight_map'])==502 and len(set(index['weight_map'].values()))==502 and len(report['router_biases_fp32'])==26 and report['source_frozen_fields']==104 and report['protected_rows_per_tensor']==256
 heads=ready['all509_RNO_object_HEADs_match_worker_streamed_full_SHA_manifest'];assert len(heads)==len(manifest['objects'])==509 and {x['uri'] for x in heads}=={t['location']+'/'+x['name']for x in manifest['objects']};assert all(x['bytes']>0 and x['actual_object_name'].startswith('marin-rno2a/')for x in heads)
 assert ready['independent_global_and_subset_aggregate_replay_EXACT']is True and ready['frozen104_output_hashes_exact_v01_source']is True
 orig=json.loads((B/'A11-BFCL-five-r1/launch-plan.json').read_text())['targets']['A11'];a=yaml.safe_load(Path(orig['model_config']).read_text());b=yaml.safe_load(Path(t['model_config']).read_text());assert {k:v for k,v in a.items()if k not in ['name','location']}=={k:v for k,v in b.items()if k not in ['name','location']}
 qs=importlib.util.spec_from_file_location('qualification',H/'check_qualification.py');qm=importlib.util.module_from_spec(qs);qs.loader.exec_module(qm);qualified=qm.check(p)
 spec=importlib.util.spec_from_file_location('packet',H/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 original_proof=B/'A11-BFCL-five-r1/native-five-bindings-proof.json';cached=json.loads(original_proof.read_text())['all5'][0]['task_config']['config'];tax=cached['error_taxonomy'];cached['error_taxonomy']=HarborErrorTaxonomy(**{k:(v if k=='commit'else frozenset(ast.literal_eval(v[len('frozenset('):-1])))for k,v in tax.items()});bm=BenchmarkMetadataRef(schema_version=1,task='bfcl_parity',primary_metric='reward',metric_kind=MetricKind.CONTINUOUS,metrics=(BenchmarkMetricRef(name='reward',source_name='reward',kind=MetricKind.CONTINUOUS,higher_is_better=True),),n_benchmark=123,n_attempted=123);assert str(bm)==cached['benchmark'];cached.update(benchmark=bm,dataset_kind=HarborDatasetKind(cached['dataset_kind']),verifier_env_keys=tuple(cached['verifier_env_keys']));fixture=ValidatedHarborConfig(**cached)
 oldpre=launch.preflight_harbor_configs;oldresolve=held_bfcl.resolve_secret_spec;oldenv=held_bfcl.env_vars_from_keys
 def preflight(requests,**kw):assert all(Path(x).read_bytes()==Path(p['suite']['path']).read_bytes()for x,_ in requests);return [fixture for _ in requests]
 launch.preflight_harbor_configs=preflight;held_bfcl.resolve_secret_spec=lambda _:SimpleNamespace(value='fixture-only');held_bfcl.env_vars_from_keys=lambda _:dict(MARIN_PREFIX=os.environ['MARIN_PREFIX']);proofs=[];dirs=set()
 class Capture:
  def submit(self,**kw):
   inspect.signature(IrisClient.submit).bind(None,**kw);assert kw['timeout'].to_seconds()==176400 and kw['max_retries_failure']==kw['max_retries_preemption']==0 and kw['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION;assert kw['environment'].env_vars['MARIN_PREFIX']==os.environ['MARIN_PREFIX'];fn,args,k=kw['entrypoint'].resolve();assert fn is held_bfcl.run_batch and not k;self.batch=args[0];return SimpleNamespace(job_id='fixture-only')
 try:
  for n in [1, 2, 5]:
   batch=m.prepare(label,n);run=next(x for x in t['runs']if x['run']==n);assert batch.priority_band==job_pb2.PRIORITY_BAND_PRODUCTION and batch.records_prefix==run['records_prefix']and batch.model.location==t['location'];e=batch.evaluations[0];assert e.identity.output_dir.startswith(run['records_prefix']+'/')and e.identity.output_dir not in dirs;dirs.add(e.identity.output_dir);assert json.loads(e.executor.config.stable_policy_json)==json.loads(fixture.stable_policy_json)
   serve=held_bfcl.inference_config_for_model(batch.model,batch.accelerator,env_vars={},priority=batch.priority_band,capability_origin=batch.capability_origin,api_model=batch.api_model);assert serve.iris.priority==job_pb2.PRIORITY_BAND_PRODUCTION;c=Capture();m.submit(batch,c);assert c.batch==batch;proofs.append(dict(run=n,records_prefix=run['records_prefix'],output_dir=e.identity.output_dir,task_config=asdict(e.executor),priority_root_and_serve='production'))
 finally:launch.preflight_harbor_configs=oldpre;held_bfcl.resolve_secret_spec=oldresolve;held_bfcl.env_vars_from_keys=oldenv
 z=create_workspace_zip(W);assert hashlib.sha256(z).hexdigest()=='5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b'
 out=dict(result='PASS',code_pin=p['code_pin'],native_bundle_sha256=hashlib.sha256(z).hexdigest(),native_bundle_bytes=len(z),actual_readiness_sha256=p['actual_readiness']['sha256'],actual_merge_metadata_and_RNO_landing_502_index_joins=True,unchanged_model_profile_except_name_location=True,qualified_native_originals=qualified,all_replacements=proofs,preflight_scope='Replay exact sealed prior validated fixed123 config at proof boundary; actual prepare still performs standard native preflight.',remote_calls=0)
 (H/'native-replacement-bindings-proof.json').write_text(json.dumps(out,indent=2,default=str)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['all_replacements','qualified_native_originals']}))
if __name__=='__main__':main()
