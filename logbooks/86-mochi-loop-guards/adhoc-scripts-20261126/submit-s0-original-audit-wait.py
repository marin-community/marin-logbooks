import subprocess
program=r'''
import json,subprocess,sys,time
from datetime import datetime,timezone
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
configure_coreweave_s3()
run_output='s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-f28d12bf4c-in-af5ab9b8/2026.11.25.9'
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9'
pattern=StoragePath(run_output)/'exports/**/dumped_evals/*/eval_only/*swesmith*.jsonl'
started=time.monotonic();last_logged=-570
while not list(pattern.glob()):
    elapsed=time.monotonic()-started
    if elapsed-last_logged>=570:
        print(json.dumps({'phase':'waiting_for_final_swesmith_dump','elapsed_seconds':round(elapsed),'no_model_trials':True}),flush=True);last_logged=elapsed
    time.sleep(30)
print(json.dumps({'phase':'full_original_audit_starting','observed_at':datetime.now(timezone.utc).isoformat(),'no_model_trials':True}),flush=True)
subprocess.run([sys.executable,'-m','experiments.post_training.mochi_loop_guards.audit','--run-output',run_output,'--output',output],check=True)
StoragePath(output+'/original-run-audit-receipt.json').write_text(json.dumps({'original_run_audit_completed':True,'baseline_sealed':False,'pending_registered_tb2_repair':True,'completed_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
'''
compile(program,'full-original-s0-audit-wait','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','16','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','28800','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-audit-20261125a','-e','MARIN_PREFIX','s3://marin-us-east-08a/marin','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(args).returncode)
