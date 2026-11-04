from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;B=HERE.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'launch-plan.json').read_text());root=Path(plan['workspace']);proof=json.loads((HERE/'native-recovery-bindings-proof.json').read_text());lifecycle=json.loads((HERE/'local-recovery-lifecycle-proof.json').read_text())
assert proof['result']==lifecycle['result']=='PASS' and len(lifecycle['cases'])==5
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
files=[p for p in HERE.iterdir() if p.is_file() and p.name!='review-receipt.json']
files += [root/'experiments/post_training/identity_path_b'/name for name in ['write_recovery.py','write_recovery_binding.json','ablation.py','ablation_launch.py','async_launch.py','checkpoints.py','export_preflight.py','storage_audit.py','worker.py','rehydrate.py','workflow.py']]
files += [Path(plan['runtime']['local'])]
for label in ['A9','A10']:
 files += [B/f'ablation-{label.lower()}-r1'/name for name in ['launch-plan.json','review-receipt.json','science-verdict-marten-r1.json','stage-files.json']]
 stage=json.loads((B/f'ablation-{label.lower()}-r1/stage-files.json').read_text());files += [Path(x['local']) for x in stage['files']]
index=B/'ablation-a9-r1/original-native71-recovery-evidence-margay-r1/actual-three-metadata-fullget-index.json';files.append(index);files += [Path(x['path']) for x in json.loads(index.read_text())['metadata']]
files += [B/'native-boundary-recovered-a9-a10-r1/launch-plan.json',B/'native-boundary-a9-a10-r1/science-verdict-marten-r1.json']
assert len(files)==len(set(files))
out=dict(schema_version=1,code_pin=plan['code_pin'],plan_sha256=sha(HERE/'launch-plan.json'),native_bundle_sha256=proof['native_bundle_sha256'],status='LOCAL_PROOFS_PASS_OWNER_RECOVERY_AUTHORITY_STORAGE_WRITE_GATE_REQUIRED',science_data_recipe_unchanged=True,review_waiver='Owner recovery ruling: no repeat science review unless scientific exporter/audit method changes; exact scientific workers carried',actual_original_A9_metadata_verified=True,actual_recovered_HF_readiness_claimed=False,application_submits_by_genet=0,files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
(HERE/'review-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(sha(HERE/'review-receipt.json'),len(files))
