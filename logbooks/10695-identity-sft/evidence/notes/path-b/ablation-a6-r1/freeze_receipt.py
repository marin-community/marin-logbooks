"""Seal the final one-arm A6 data/recipe/native binding and staging join."""
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/karan/oa/marin-path-b-a6-glm-off');B=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'launch-plan.json').read_text());pin=plan['code_pin']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==pin
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
stage=json.loads((HERE/'stage-files.json').read_text());receipt=json.loads((HERE/'three-inputs-staging-FULLGET-readback-margay-r1.json').read_text())
assert receipt['manifest_sha256']==sha(HERE/'stage-files.json')
for source,remote in zip(stage['files'],receipt['files'],strict=True):
 assert all(source[k]==remote[k] for k in ['local','uri','sha256','bytes']) and remote['FULLGET_equal'] is True
 assert sha(Path(source['local']))==source['sha256'] and Path(source['local']).stat().st_size==source['bytes']
files=[HERE/name for name in ['PACKET.md','launch-plan.json','stage-files.json','three-inputs-staging-FULLGET-readback-margay-r1.json','native-A6-bindings-proof.json','all-native-and-recipe-proof.json','data-counts.json','untruncated-A0-prefix-census.json','ON-decisions-untruncated.jsonl','build_data.py','census_prefix.py','build_plan.py','check_bindings.py','prove_native_and_recipe.py','freeze_receipt.py','prepare_request.py','model-A6.yaml','build-data.log','native-bindings.log','native-and-recipe.log','lint.log','types.log']]
files += [Path(row['local']) for row in stage['files']]
files += [ROOT/'experiments/post_training/identity_path_b'/name for name in ['ablation.py','ablation_launch.py','ablation_data.py','trace_schedule.py','recipe.py','worker.py','storage_audit.py','retention.py','export_preflight.py']]
files += [Path(plan['runtime']['local']),B/'bfcl-ablation-r1/launch-plan.json',B/'bfcl-ablation-r1/independent-data-proof-marten-r1.json',B/'glm-traces-pool-r1/revision-r2/science-verdict-marten.json',B/'r5-refusal-cap-r1/source-census-and-selection.json']
result=dict(schema_version=1,code_pin=pin,parent_pin='6ac2c53d061156ad22be53b332b7bd677a856290',changed_source_files=['experiments/post_training/identity_path_b/ablation.py'],source_delta='one enum value only; same optimizer/freezes/audit/lifecycle',plan_sha256=sha(HERE/'launch-plan.json'),input_manifest_sha256=sha(HERE/'stage-files.json'),staging_receipt_sha256=sha(HERE/'three-inputs-staging-FULLGET-readback-margay-r1.json'),status='FINAL_LOCAL_PROOFS_PASS_MARTEN_COMBINED_READ_PENDING',submit_authorized=False,remote_calls_by_genet=0,files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files})
(HERE/'review-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(sha(HERE/'review-receipt.json'),len(result['files']))
