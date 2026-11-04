from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'launch-plan.json').read_text());root=Path(plan['workspace'])
files=list(HERE.glob('*.py'))+[HERE/'README.md',HERE/'launch-plan.json',HERE/'test-results.json',HERE/'lint-types-final.log']
if (HERE/'native-binding-proof.json').exists(): files.append(HERE/'native-binding-proof.json')
files += [Path(plan[k]['path']) for k in ['original_plan','historical_failure','original_native_receipt','source_preservation_receipt']]
files += [Path(plan['runtime']['local'])]
files += [root/'experiments/post_training/identity_path_b'/n for n in ['ablation_recovery.py','test_ablation_recovery.py','ablation.py','ablation_launch.py','async_launch.py','workflow.py','export_preflight.py','worker.py','storage_audit.py','checkpoints.py']]
files += [Path(plan['bfcl']['model_config']),Path(plan['bfcl']['policy_wrapper'])]
rows=[]
for p in sorted(set(files)):
 raw=p.read_bytes();rows.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
(HERE/'receipt.json').write_text(json.dumps(dict(code_pin=plan['code_pin'],original_training_pin=plan['original_training_pin'],plan_sha256=hashlib.sha256((HERE/'launch-plan.json').read_bytes()).hexdigest(),files=rows,remote_calls_by_genet=0),indent=2)+'\n')
print(len(rows),hashlib.sha256((HERE/'receipt.json').read_bytes()).hexdigest())
