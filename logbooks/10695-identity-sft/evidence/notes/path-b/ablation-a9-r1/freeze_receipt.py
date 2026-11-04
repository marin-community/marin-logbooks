"""Seal A9 source/data/native bindings and current regional readback."""
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;B=HERE.parent;plan=json.loads((HERE/'launch-plan.json').read_text());ROOT=Path(plan['workspace']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==plan['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
stage=json.loads((HERE/'stage-files.json').read_text());remote=json.loads((HERE/'three-inputs-staging-FULLGET-readback-margay-r1.json').read_text());assert remote['manifest_sha256']==sha(HERE/'stage-files.json');assert len(stage['files'])==len(remote['files'])==3
for expected,actual in zip(stage['files'],remote['files'],strict=True):
 assert all(expected[k]==actual[k] for k in ['local','uri','bytes','sha256'])
 assert actual['remote_bytes']==expected['bytes'] and actual['remote_sha256']==expected['sha256'] and actual['state']=='FULL_REMOTE_GET_SHA_LENGTH_PASS'
 assert Path(expected['local']).stat().st_size==expected['bytes'] and sha(Path(expected['local']))==expected['sha256']
files=[p for p in HERE.iterdir() if p.is_file() and p.name not in ['review-receipt.json','science-verdict-marten-r1.json','science-reading-marten-r1.md']]
files += [Path(x['local']) for x in stage['files']]+[Path(stage['files'][0]['local']).parent/'sequence-rows.jsonl']
files += [ROOT/'experiments/post_training/identity_path_b'/name for name in ['ablation.py','ablation_data.py','ablation_launch.py','test_ablation.py','test_ablation_launch.py','trace_schedule.py','recipe.py','worker.py','storage_audit.py','retention.py','export_preflight.py']]
files += [Path(plan['runtime']['local']),B/'ablation-a8-r1/launch-plan.json',B/'ablation-a8-r1/review-receipt.json',B/'r5-refusal-cap-r1/source-census-and-selection.json',B/'bfcl-noise-r1/A9-run-comparison-preregistration-marten-r1.json']
assert len(files)==len(set(files))
receipt=dict(schema_version=1,code_pin=plan['code_pin'],parent_pin='4bfe87adbb03372d9b51ff4c87b8f3a637e54bfe',source_delta='A9 arm-specific72 horizon/final71 with old36 defaults; allother optimizer/precision/freezes/audits/lifecycle unchanged',plan_sha256=sha(HERE/'launch-plan.json'),input_manifest_sha256=sha(HERE/'stage-files.json'),staging_receipt_sha256=sha(HERE/'three-inputs-staging-FULLGET-readback-margay-r1.json'),status='AUTHOR_LOCAL_PROOFS_PASS_INDEPENDENT_COMBINED_READ_PENDING',submit_authorized=False,remote_calls_by_genet=0,files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files})
(HERE/'review-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(sha(HERE/'review-receipt.json'),len(files))
