# Original-run diagnostic only; final S0 audit must apply the registered TB2 recovery input.
import hashlib
import json
from pathlib import Path
import subprocess

state = json.loads(Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261125_1835_s0_v9_monitoring_state.json').read_text())
source = Path('/tmp/mochi-loop-guards/s0-counterfactual-regional.py').read_text()
compile(source, 'regional-counterfactual', 'exec')
output = 's3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/original-counterfactual-diagnostic'
program = '\n'.join([
    'import hashlib, json, subprocess, sys',
    'from marin.rl.skyrl import skyrl_cpu_command',
    'from rigging.filesystem.s3_compat import configure_coreweave_s3',
    'from rigging.filesystem.storage_path import StoragePath',
    'configure_coreweave_s3()',
    f'run_output={state["output"]!r}',
    f'output={output!r}',
    f'analysis_source={source!r}',
    '(StoragePath(output)/"regional-counterfactual.py").write_text(analysis_source)',
    '(StoragePath(output)/"resolved-manifest.json").write_bytes(StoragePath("s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/resolved-manifest.json").read_bytes())',
    'subprocess.run(skyrl_cpu_command("python","-c",analysis_source,"--manifest",output+"/resolved-manifest.json","--pattern",run_output+"/exports/**/dumped_evals/*/eval_only/*.jsonl","--output",output+"/counterfactual"),check=True)',
    '(StoragePath(output)/"postprocessing-receipt.json").write_text(json.dumps({"source_sha256":hashlib.sha256(analysis_source.encode()).hexdigest(),"counterfactual_diagnostic_success":True,"baseline_sealed":False,"requires_canonical_recovery_recomputation":True},indent=2)+"\\n")',
])
compile(program, 'regional-postprocessing', 'exec')
args = ['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','16','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','14400','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-original-cf-20261126a','-e','MARIN_PREFIX','s3://marin-us-east-08a/marin','--','uv','run','--package','marin-core','python','-c',program]
if __name__ == '__main__':
    import sys
    if sys.argv[1:] != ['--submit']:
        print(json.dumps({'validated':True,'gpu_submitted':False,'run_output':state['output'],'output':output,'analysis_sha256':hashlib.sha256(source.encode()).hexdigest()}))
    else:
        raise SystemExit(subprocess.run(args).returncode)
