# Corrected full S0 v12 only. Submit after all four dumps and terminal GPU/root state.
import hashlib
import json
from pathlib import Path
import subprocess

state = json.loads(Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261126_0050_s0_v12_monitoring_state.json').read_text())
source = Path('/tmp/mochi-loop-guards/s0-counterfactual-regional.py').read_text()
census_source = Path('/tmp/mochi-loop-guards/s0-v12-observation-census.py').read_text()
compile(census_source, 'regional-observation-census', 'exec')
compile(source, 'regional-counterfactual', 'exec')
output = 's3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12'
program = '\n'.join([
    'import hashlib, json, subprocess, sys, yaml',
    'from marin.rl.skyrl import skyrl_cpu_command',
    'from rigging.filesystem.s3_compat import configure_coreweave_s3',
    'from rigging.filesystem.storage_path import StoragePath',
    'configure_coreweave_s3()',
    f'run_output={state["output"]!r}',
    f'output={output!r}',
    f'analysis_source={source!r}',
    '(StoragePath(output)/"regional-counterfactual.py").write_text(analysis_source)',
    'subprocess.run([sys.executable,"-m","experiments.post_training.mochi_loop_guards.audit","--run-output",run_output,"--output",output],check=True)',
    'subprocess.run(skyrl_cpu_command("python","-c",analysis_source,"--manifest",output+"/resolved-manifest.json","--pattern",run_output+"/exports/**/dumped_evals/*/eval_only/*.jsonl","--output",output+"/counterfactual"),check=True)',
    f'census_source={census_source!r}',
    '(StoragePath(output)/"regional-observation-census.py").write_text(census_source)',
    'subprocess.run(skyrl_cpu_command("python","-c",census_source,"--manifest",output+"/resolved-manifest.json","--pattern",run_output+"/exports/**/dumped_evals/*/eval_only/*.jsonl","--output",output+"/observation-census.json"),check=True)',
    'native=yaml.safe_load((StoragePath(run_output)/"resolved-launch.yaml").read_text())["config"]',
    'retention_source=native["skyrl"]["generator"]["trajectory_retention"]["output_path"]',
    'retention_destination="s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/traces/s0-v12"',
    'assert retention_source.startswith("s3://marin-us-east-08a/")',
    'subprocess.run(["uv","run","--package","marin-core","fsutil","verified-copy",retention_source,retention_destination,"--workers","8"],check=True)',
    'retention_manifest=(StoragePath(retention_destination)/".verified-copy-manifest.json").read_bytes()',
    '(StoragePath(output)/"retention-promotion.json").write_text(json.dumps({"source":retention_source,"destination":retention_destination,"completion_manifest":retention_destination+"/.verified-copy-manifest.json","manifest_sha256":hashlib.sha256(retention_manifest).hexdigest(),"same_region":True},indent=2)+"\\n")',
    '(StoragePath(output)/"postprocessing-receipt.json").write_text(json.dumps({"source_sha256":hashlib.sha256(analysis_source.encode()).hexdigest(),"audit_and_counterfactual_success":True,"baseline_sealed":False,"requires_per_cohort_validity_review":True,"requires_i1_precision_review":True},indent=2)+"\\n")',
])
compile(program, 'regional-postprocessing', 'exec')
args = ['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','16','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','14400','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-postprocess-20261126v12a','-e','MARIN_PREFIX','s3://marin-us-east-08a/marin','--','uv','run','--package','marin-core','python','-c',program]
if __name__ == '__main__':
    import sys
    if sys.argv[1:] != ['--submit']:
        print(json.dumps({'validated':True,'gpu_submitted':False,'run_output':state['output'],'output':output,'analysis_sha256':hashlib.sha256(source.encode()).hexdigest()}))
    else:
        # The shared trajectory sink must finish publication before its prefix is snapshotted.
        for kind in ("coordinator_job_id", "gpu_job_id"):
            terminal = subprocess.run(
                ["uv", "run", "--no-sync", "iris", "--cluster", "marin", "job", "describe", state[kind]],
                capture_output=True, text=True, check=True,
            )
            status = next(line for line in terminal.stdout.splitlines() if line.startswith("State:"))
            if status.split()[1] not in ("succeeded", "failed", "killed"):
                raise SystemExit("Wait for terminal " + kind + " before copying the shared trajectory sink")
            Path("/tmp/mochi-loop-guards/s0-v12-pre-postprocess-" + kind + ".log").write_text(
                terminal.stdout.split("\nStepSpec(")[0]
            )
        raise SystemExit(subprocess.run(args).returncode)
