from pathlib import Path
from dataclasses import asdict
import ast,hashlib,importlib.util,inspect,io,json,subprocess,zipfile
from types import SimpleNamespace
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from experiments.evaluation import held_bfcl
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((HERE/'launch-plan.json').read_text());root=Path(plan['workspace'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    old=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing/experiments/evaluation/held_bfcl.py');new=root/'experiments/evaluation/held_bfcl.py'
    assert [ast.dump(n,include_attributes=False) for n in ast.parse(old.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]==[ast.dump(n,include_attributes=False) for n in ast.parse(new.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    spec=importlib.util.spec_from_file_location('swe_packet',HERE/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    oldresolver=held_bfcl.resolve_secret_spec;oldenv=held_bfcl.env_vars_from_keys
    held_bfcl.resolve_secret_spec=lambda value:SimpleNamespace(value='fixture-only')
    held_bfcl.env_vars_from_keys=lambda keys:{}
    proofs=[];outputs=set()
    class Capture:
        def submit(self,**kwargs):
            inspect.signature(IrisClient.submit).bind(None,**kwargs)
            assert kwargs['timeout'].to_seconds()==176400
            assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0
            fn,args,kw=kwargs['entrypoint'].resolve();assert fn is held_bfcl.run_batch and not kw
            self.batch=args[0];return SimpleNamespace(job_id='fixture-only')
    try:
        for label,t in plan['targets'].items():
            for run in range(1,4):
                batch=m.prepare(label,run)
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
        held_bfcl.resolve_secret_spec=oldresolver;held_bfcl.env_vars_from_keys=oldenv
    bundle=create_workspace_zip(root)
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        for name in ['experiments/evaluation/held_bfcl.py','config/coreweave.yaml','experiments/evaluation/task238-screens/swebench-verified-100.yaml']:
            assert z.read(name)==(root/name).read_bytes()
        members=len(z.namelist())
    out={'code_pin':plan['code_pin'],'bundle_sha256':hashlib.sha256(bundle).hexdigest(),'bundle_bytes':len(bundle),'members':members,'all12':proofs,'function_AST_equal_exercised_wrapper':True,'calls_to_model_network_or_storage':0}
    (HERE/'native-twelve-bindings-proof.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='all12'}))
if __name__=='__main__':main()
