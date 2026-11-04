from pathlib import Path
from dataclasses import asdict
import ast,hashlib,importlib.util,inspect,io,json,os,subprocess,zipfile
from types import SimpleNamespace
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from experiments.evaluation import held_bfcl
from experiments.evaluation import launch
from marin.evaluation.harbor.driver_config import ValidatedHarborConfig,HarborDatasetKind,HarborErrorTaxonomy
from marin.evaluation.records import BenchmarkMetadataRef,BenchmarkMetricRef,MetricKind
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((HERE/'launch-plan.json').read_text());root=Path(plan['workspace'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    old=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing/experiments/evaluation/held_bfcl.py');new=root/'experiments/evaluation/held_bfcl.py'
    assert [ast.dump(n,include_attributes=False) for n in ast.parse(old.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]==[ast.dump(n,include_attributes=False) for n in ast.parse(new.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    spec=importlib.util.spec_from_file_location('swe_packet',HERE/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    parent_spec=importlib.util.spec_from_file_location('parent_swe_packet',HERE.parent/'second-agentic-SWE100-r1/prepare_request.py');parent=importlib.util.module_from_spec(parent_spec);parent_spec.loader.exec_module(parent)
    sealed=json.loads((HERE.parent/'second-agentic-SWE100-r1/native-twelve-bindings-proof.json').read_text())['all12']
    cached=dict(sealed[0]['task_config']['config'])
    assert all(x['task_config']['config']['stable_policy_json']==cached['stable_policy_json'] for x in sealed)
    taxonomy=cached['error_taxonomy'];cached['error_taxonomy']=HarborErrorTaxonomy(**{k:(v if k=='commit' else frozenset(ast.literal_eval(v[len('frozenset('):-1]))) for k,v in taxonomy.items()})
    benchmark=BenchmarkMetadataRef(schema_version=1,task='swebench-verified',primary_metric='reward',metric_kind=MetricKind.CONTINUOUS,metrics=(BenchmarkMetricRef(name='reward',source_name='reward',kind=MetricKind.CONTINUOUS,higher_is_better=True),),n_benchmark=100,n_attempted=100)
    assert str(benchmark)==cached['benchmark'];cached['benchmark']=benchmark;cached['dataset_kind']=HarborDatasetKind(cached['dataset_kind']);cached['verifier_env_keys']=tuple(cached['verifier_env_keys'])
    config_fixture=ValidatedHarborConfig(**cached);original_preflight=launch.preflight_harbor_configs
    def replay_preflight(requests,**kwargs):
        assert all(Path(path).read_bytes()==Path(plan['suite']['path']).read_bytes() for path,_ in requests)
        return [config_fixture for _ in requests]
    launch.preflight_harbor_configs=replay_preflight
    oldresolver=held_bfcl.resolve_secret_spec;oldenv=held_bfcl.env_vars_from_keys
    held_bfcl.resolve_secret_spec=lambda value:SimpleNamespace(value='fixture-only')
    held_bfcl.env_vars_from_keys=lambda keys:{k:os.environ[k] for k in keys if k=='MARIN_PREFIX' and k in os.environ}
    proofs=[];outputs=set()
    class Capture:
        def submit(self,**kwargs):
            inspect.signature(IrisClient.submit).bind(None,**kwargs)
            assert kwargs['timeout'].to_seconds()==176400
            assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0
            assert kwargs['environment'].env_vars['MARIN_PREFIX']==os.environ['MARIN_PREFIX']
            fn,args,kw=kwargs['entrypoint'].resolve();assert fn is held_bfcl.run_batch and not kw
            self.batch=args[0];return SimpleNamespace(job_id='fixture-only')
    try:
        for label,t in plan['targets'].items():
            for item in t['runs']:
                run=item['run']
                oldbatch=parent.prepare(label,run)
                batch=m.prepare(label,run)
                item=next(x for x in t['runs'] if x['run']==run)
                assert batch.records_prefix==item['records_prefix'] and os.environ['MARIN_PREFIX']==item['MARIN_PREFIX']
                assert batch.model==oldbatch.model and batch.evaluations[0].executor.config.stable_policy_json==oldbatch.evaluations[0].executor.config.stable_policy_json
                assert not batch.records_prefix.startswith(item['original_records_prefix']+'/') and item['original_records_prefix']!=batch.records_prefix
                assert batch.model.location==t['location'] and len(batch.evaluations)==1
                e=batch.evaluations[0];cfg=e.executor.config
                assert e.identity.output_dir.startswith(batch.records_prefix+'/')
                assert e.identity.output_dir not in outputs;outputs.add(e.identity.output_dir)
                policy=json.loads(cfg.stable_policy_json)
                assert policy['datasets'][0]['task_names']==plan['suite']['task_ids'] and policy['n_attempts']==1
                assert policy['agents'][0]['kwargs']['version']=='2.1.0'
                assert policy['retry']['max_retries']==10
                print('CONFIG_TYPE',type(cfg).__name__,flush=True) if not proofs else None
                # Native serialized Harbor config is the actual evaluator boundary.
                native=asdict(e.executor)
                assert batch.accelerator.target_cluster=='cw-rno2a' and batch.judge is None
                client=Capture();m.submit(batch,client);assert client.batch==batch
                proofs.append({'label':label,'run':run,'model':batch.model.location,'task_config':native,'output_dir':e.identity.output_dir,'native_callable_roundtrip':'PASS','timeout_seconds':176400,'nested_serve_seconds':43200,'records_prefix':batch.records_prefix})
    finally:
        held_bfcl.resolve_secret_spec=oldresolver;held_bfcl.env_vars_from_keys=oldenv;launch.preflight_harbor_configs=original_preflight
    bundle=create_workspace_zip(root)
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        for name in ['experiments/evaluation/held_bfcl.py','config/coreweave.yaml','experiments/evaluation/task238-screens/swebench-verified-100.yaml']:
            assert z.read(name)==(root/name).read_bytes()
        members=len(z.namelist())
    assert hashlib.sha256(bundle).hexdigest()=='5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b'
    out={'code_pin':plan['code_pin'],'bundle_sha256':hashlib.sha256(bundle).hexdigest(),'bundle_bytes':len(bundle),'members':members,'all4':proofs,'function_AST_equal_exercised_wrapper':True,'calls_to_model_network_or_storage':0,'Harbor_preflight_proof_scope':'Exact sealed original validated config replay at subprocess boundary; source production prepare/submit is unchanged and still does normal preflight. Initial fresh local driver wait was interrupted without model or submit calls.'}
    (HERE/'native-four-bindings-proof.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='all4'}))
if __name__=='__main__':main()
