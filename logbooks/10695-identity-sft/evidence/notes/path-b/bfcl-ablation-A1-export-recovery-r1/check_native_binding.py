from pathlib import Path
import hashlib,inspect,json,subprocess
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint
from experiments.post_training.identity_path_b.ablation import export_worker
from experiments.post_training.identity_path_b.ablation_recovery import export_only_worker
from experiments.post_training.identity_path_b.checkpoints import event_bytes
from experiments.post_training.identity_path_b.workflow import EXPORT_RESOURCES,EXPORT_LIFETIME
from fray.iris_backend import convert_resources
from prepare_request import request
HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'launch-plan.json').read_text())
assert not subprocess.check_output(['git','status','--porcelain'],cwd=plan['workspace'],text=True).strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan['workspace'],text=True).strip()==plan['code_pin']
changed=subprocess.check_output(['git','diff','--name-only',plan['original_training_pin'],plan['code_pin']],cwd=plan['workspace'],text=True).splitlines()
assert set(changed)=={'experiments/post_training/identity_path_b/'+n for n in ['ablation_recovery.py','test_ablation_recovery.py']}
req=request({'CW_KEY_ID':'fixture-only','CW_KEY_SECRET':'fixture-only'})
inspect.signature(IrisClient.submit).bind(None,**req)
fn,args,kwargs=req['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kwargs)
assert fn is export_only_worker and not args
assert kwargs['training']['training_pin']==plan['original_training_pin']
event=kwargs['event'];assert hashlib.sha256(event_bytes(event)).hexdigest()==plan['evidence']['event_sha256']=='6e676cd4005e6a4f20332e50aae5e72696ef6a8c3b55c2b5984c27989633d2f4'
assert req['timeout'].to_seconds()==176400 and req['max_retries_failure']==req['max_retries_preemption']==0
assert set(req['environment'].secrets)=={'CW_KEY_ID','CW_KEY_SECRET','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY'}
assert req['environment'].env_vars['JAX_PLATFORMS']=='cpu'
export_kwargs={k:kwargs['training'][k] for k in ['run_id','runtime_bytes','runtime_sha256','source_receipt','source_objects']}
export=Entrypoint.from_callable(export_worker,event=event,**export_kwargs)
f,a,k=export.resolve();inspect.signature(f).bind(*a,**k);assert f is export_worker and k['event']==event
resource=convert_resources(EXPORT_RESOURCES).to_proto();assert resource.device.gpu.count==8 and EXPORT_LIFETIME==7200
out=dict(code_pin=plan['code_pin'],original_training_pin=plan['original_training_pin'],original_event_sha256=plan['evidence']['event_sha256'],native_submit_signature='PASS',actual_CPU_export_only_entrypoint='PASS',actual_unchanged_export_kwargs='PASS',only_two_new_files_vs6ac='PASS',no_train_in_resolved_owner='PASS',GPU_count=8,export_timeout=7200,root_timeout=176400,requested_retries=[0,0],remote_calls=0)
(HERE/'native-binding-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
