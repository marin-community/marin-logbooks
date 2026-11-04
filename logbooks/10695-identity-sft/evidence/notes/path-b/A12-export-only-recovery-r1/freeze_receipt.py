from pathlib import Path
import hashlib,json,subprocess
H=Path(__file__).resolve().parent;B=H.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((H/'launch-plan.json').read_text());W=Path(plan['workspace']);proof=json.loads((H/'native-binding-proof.json').read_text());life=json.loads((H/'local-recovery-lifecycle-proof.json').read_text());assert proof['result']==life['result']=='PASS';assert len(life['cases'])==5;assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==plan['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
assert sha(H/'author-sealed-workspace.zip')==proof['native_bundle_sha256']
files=[p for p in H.iterdir() if p.is_file() and p.name not in ['review-receipt.json','author-sealed-workspace.zip']]
files += [W/'experiments/post_training/identity_path_b'/n for n in ['a12_export_recovery.py','a12_export_binding.json','ablation.py','ablation_launch.py','async_launch.py','checkpoints.py','export_preflight.py','worker.py','storage_audit.py','workflow.py','rehydrate.py']]
files += [Path(plan['runtime']['local'])]
files += [B/'ablation-a12-a13-r1'/n for n in ['launch-plan.json','review-receipt.json','combined-review-marten-r1.json','stage-files.json','six-inputs-staging-FULLGET-readback-margay-r1.json']]
files += [B/'ablation-a12-a13-r1/A12'/n for n in ['actual-trained18-native17-export-recovery-source-proof-margay-r1.json','terminal-small-trained-native-event-audit-FULLGET-margay-r1.json']]
files += [Path(p) for p in json.loads((H/'local-evidence-map.json').read_text()).values()]
files += [Path(x['local']) for x in json.loads((B/'ablation-a12-a13-r1/stage-files.json').read_text())['files'] if '/A12/' in x['uri']]
assert len(files)==len(set(files))
receipt=dict(schema_version=1,code_pin=plan['code_pin'],parent_scientific_pin=plan['parent_pin'],plan_sha256=sha(H/'launch-plan.json'),native_bundle_sha256=proof['native_bundle_sha256'],native_bundle_bytes=proof['native_bundle_bytes'],status='AUTHOR_LOCAL_BINDING_AND_LIFECYCLE_PASS_NARROW_REVIEW_PENDING',priority_owner='production',priority_export='production',training_children=0,only_owned_original_native17_retired_after_full_fresh502_audit=True,actual_recovered_HF_readiness_claimed=False,application_calls_by_genet=0,files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
(H/'review-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(receipt_sha256=sha(H/'review-receipt.json'),files=len(files),plan_sha256=receipt['plan_sha256'],source=plan['code_pin'],bundle=proof['native_bundle_sha256'])))
