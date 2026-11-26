from pathlib import Path
import hashlib,subprocess
source=Path('/tmp/mochi-loop-guards/stitch-tb2-recovery-regional.py').read_text()
compile(source,'tb2-stitcher','exec')
source_sha=hashlib.sha256(source.encode()).hexdigest()
run='s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-f28d12bf4c-in-af5ab9b8/2026.11.25.9'
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/tb2-stitch-selfcheck'
audit='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/partial/tb2'
receipt='s3://marin-us-east-08a/marin/rl-data/task-selection/users/benfeuer/mochi-loop-guards/s0-infra-recovery-tb2/2026.11.25.1/recovery-receipt.json'
program='\n'.join([
 'import json,subprocess',
 'from marin.rl.skyrl import skyrl_cpu_command',
 'from rigging.filesystem.s3_compat import configure_coreweave_s3',
 'from rigging.filesystem.storage_path import StoragePath',
 'configure_coreweave_s3()',
 f'source={source!r}',f'output={output!r}',
 '(StoragePath(output)/"stitcher.py").write_text(source)',
 f'subprocess.run(skyrl_cpu_command("python","-c",source,"--original-run-output",{run!r},"--recovery-run-output",{run!r},"--recovery-receipt",{receipt!r},"--output",output,"--source-sha256",{source_sha!r}),check=True)',
 f'subprocess.run(skyrl_cpu_command("python","vendor/skyrl/scripts/loop_audit.py","--manifest",{audit+"/resolved-manifest.json"!r},"--input","tb2="+output+"/DCAgent2_terminal_bench_2.jsonl","--output",output+"/audit.json"),check=True)',
 f'original=json.loads(StoragePath({audit+"/audit.json"!r}).read_text())',
 'new=json.loads((StoragePath(output)/"audit.json").read_text())',
 'assert new["domains"]==original["domains"] and new["agents"]==original["agents"] and new["cohort_hashes"]==original["cohort_hashes"]',
 'result={"selfcheck_only":True,"new_model_trials":0,"original_as_simulated_replay":True,"all_audit_metrics_equal":True,"full_s0_sealed":False}',
 '(StoragePath(output)/"selfcheck-receipt.json").write_text(json.dumps(result,indent=2)+"\\n")',
 'print(json.dumps(result))',
])
compile(program,'tb2-stitch-selfcheck','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','32GB','--disk','64GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-tb2-stitch-selfcheck-20261125a','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(args).returncode)
