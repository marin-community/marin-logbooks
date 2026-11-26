raise SystemExit("SUPERSEDED: v9 and v10 observation traces invalid; use corrected S0 v11 instead")
import argparse
import hashlib
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--recovery-run-output',required=True)
parser.add_argument('--submit',action='store_true')
args=parser.parse_args()
original='s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-f28d12bf4c-in-af5ab9b8/2026.11.25.9'
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/canonical'
selection='s3://marin-us-east-08a/marin/rl-data/task-selection/users/benfeuer/mochi-loop-guards/s0-infra-recovery-tb2/2026.11.25.1'
stitch=Path('/tmp/mochi-loop-guards/stitch-tb2-recovery-regional.py').read_text()
counterfactual=Path('/tmp/mochi-loop-guards/s0-counterfactual-regional.py').read_text()
for label,source in [('stitch',stitch),('counterfactual',counterfactual)]:compile(source,label,'exec')
program='\n'.join([
'import hashlib,json,subprocess',
'from marin.rl.skyrl import skyrl_cpu_command',
'from rigging.filesystem.s3_compat import configure_coreweave_s3',
'from rigging.filesystem.storage_path import StoragePath',
'from experiments.post_training.mochi_loop_guards.audit import resolved_manifest,REGIONAL_SELECTION',
'configure_coreweave_s3()',
f'original={original!r}',f'recovery={args.recovery_run_output!r}',f'output={output!r}',
f'stitch={stitch!r}',f'counterfactual={counterfactual!r}',
"manifest_source=StoragePath('cohorts/manifest.json').read_bytes()",
"manifest=resolved_manifest(json.loads(manifest_source),REGIONAL_SELECTION)",
"manifest['frozen_manifest_file_sha256']=hashlib.sha256(manifest_source).hexdigest()",
"manifest_path=output+'/resolved-manifest.json'",
"StoragePath(manifest_path).write_text(json.dumps(manifest,indent=2)+'\\n')",
"StoragePath(output+'/stitcher.py').write_text(stitch)",
"StoragePath(output+'/counterfactual-processor.py').write_text(counterfactual)",
f"subprocess.run(skyrl_cpu_command('python','-c',stitch,'--original-run-output',original,'--recovery-run-output',recovery,'--recovery-receipt',{selection!r}+'/recovery-receipt.json','--output',output+'/tb2','--source-sha256',hashlib.sha256(stitch.encode()).hexdigest()),check=True)",
"pattern=original+'/exports/**/dumped_evals/*/eval_only/*.jsonl'",
"canonical_tb2=output+'/tb2/DCAgent2_terminal_bench_2.jsonl'",
"command=skyrl_cpu_command('python','vendor/skyrl/scripts/loop_audit.py','--manifest',manifest_path,'--output',output+'/audit.json')",
"for cohort in manifest['cohorts']:command.extend(('--input',cohort+'='+(canonical_tb2 if cohort=='tb2' else pattern)))",
"subprocess.run(command,check=True)",
"subprocess.run(skyrl_cpu_command('python','-c',counterfactual,'--manifest',manifest_path,'--pattern',pattern,'--replacement-input','tb2='+canonical_tb2,'--output',output+'/counterfactual'),check=True)",
"receipt={'original_run_output':original,'recovery_run_output':recovery,'stitcher_sha256':hashlib.sha256(stitch.encode()).hexdigest(),'counterfactual_sha256':hashlib.sha256(counterfactual.encode()).hexdigest(),'processing_succeeded':True,'baseline_sealed':False,'requires_baseline_validity_review':True,'requires_i1_precision_review':True}",
"StoragePath(output+'/postprocessing-receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')",
])
compile(program,'regional-canonical-postprocessing','exec')
if not args.submit:
 print({'validated':True,'gpu_submitted':False,'output':output,'stitcher_sha256':hashlib.sha256(stitch.encode()).hexdigest(),'counterfactual_sha256':hashlib.sha256(counterfactual.encode()).hexdigest()})
 raise SystemExit(0)
command=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','16','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','14400','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-canonical-postprocess-20261126a','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(command).returncode)
