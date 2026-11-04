"""Seal the two A8 data/native bindings and exact regional input readback."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=HERE.parent
ROOT=Path('/home/karan/oa/marin-path-b-a8-on-share')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'launch-plan.json').read_text());pin=plan['code_pin']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==pin
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
assert {a['arm'] for a in plan['arms']}=={'A8a','A8b'}
stage=json.loads((HERE/'stage-files.json').read_text());receipt=json.loads((HERE/'six-inputs-staging-FULLGET-readback-margay-r1.json').read_text())
assert receipt['manifest_sha256']==sha(HERE/'stage-files.json')
assert len(stage['files'])==len(receipt['stage_objects'])==6
for source,remote in zip(stage['files'],receipt['stage_objects'],strict=True):
 assert all(source[k]==remote[k] for k in ['local','uri','sha256','bytes']) and remote['fresh_PUT_FULLGET_byte_equal'] is True
 assert sha(Path(source['local']))==source['sha256'] and Path(source['local']).stat().st_size==source['bytes']
files=[HERE/name for name in ['PACKET.md','launch-plan.json','stage-files.json','six-inputs-staging-FULLGET-readback-margay-r1.json','native-A8-bindings-proof.json','all-native-and-recipe-proof.json','data-counts.json','eligible-traced-occurrences-seed310.jsonl','A8a-selected-ON-occurrences.jsonl','A8b-selected-ON-occurrences.jsonl','A8a-ON-decisions.jsonl','A8b-ON-decisions.jsonl','build_data.py','build_stage_manifest.py','build_plan.py','check_bindings.py','prove_native_and_recipe.py','freeze_receipt.py','prepare_request.py','model-A8a.yaml','model-A8b.yaml','source-validation-genet-r1.json','independent-data-native-proof-marten-r1.json']]
files += [Path(row['local']) for row in stage['files']]
files += [Path(row['local']).parent/'sequence-rows.jsonl' for row in stage['files'] if row['local'].endswith('packed.parquet')]
files += [ROOT/'experiments/post_training/identity_path_b'/name for name in ['ablation.py','ablation_launch.py','ablation_data.py','trace_schedule.py','recipe.py','worker.py','storage_audit.py','retention.py','export_preflight.py']]
files += [Path(plan['runtime']['local']),B/'ablation-a6-r1/launch-plan.json',B/'ablation-a6-r1/review-receipt.json',B/'r5-refusal-cap-r1/source-census-and-selection.json',B/'bfcl-noise-r1/A8-run-comparison-preregistration-marten-r1.json',B/'bfcl-noise-r1/A8-preregistration-timing-clarification-marten-r1.json']
assert len(files)==len(set(files))
result=dict(schema_version=1,code_pin=pin,parent_pin='cc2c0160d01bde4ef38f0eb92a0e6ce4c30b9dfb',changed_source_files=['experiments/post_training/identity_path_b/ablation.py'],source_delta='two enum values only; same optimizer/freezes/audit/lifecycle',plan_sha256=sha(HERE/'launch-plan.json'),input_manifest_sha256=sha(HERE/'stage-files.json'),staging_receipt_sha256=sha(HERE/'six-inputs-staging-FULLGET-readback-margay-r1.json'),status='FINAL_LOCAL_PROOFS_PASS_MARTEN_COMBINED_READ_PENDING',submit_authorized=False,remote_calls_by_genet=0,files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files})
(HERE/'review-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(sha(HERE/'review-receipt.json'),len(files))
