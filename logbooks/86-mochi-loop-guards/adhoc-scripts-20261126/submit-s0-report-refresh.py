import subprocess
program=r'''
import json
from datetime import datetime,timezone
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from marin.rl.skyrl_records import read_stored_skyrl_attempts
from experiments.post_training import rl_compose
from experiments.post_training.mochi_loop_guards.__main__ import GUARDS
from experiments.post_training.rl_measurements import evaluation_status,read_recorded_launch
from experiments.post_training.rl_publication import build_record
configure_coreweave_s3()
rl_compose.SELECTION_ROOT='s3://marin-us-east-08a/marin/rl-data/task-selection'
run='s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-8d1a0b19d0-in-af5ab9b8/2026.11.25.9'
out=StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12')
attempt_id,attempts=read_stored_skyrl_attempts(run)
a=attempts[attempt_id]
assert a.native.result.state=='succeeded'
p=GUARDS.by_name['s0-baseline']
status=evaluation_status(run,p,a,read_recorded_launch(a))
(out/'refreshed-evaluation-status.json').write_text(status.model_dump_json(indent=2)+'\n')
summary=build_record(run,p,table=GUARDS,attempt=a)
(out/'refreshed-attempt-summary.json').write_text(summary.model_dump_json(indent=2)+'\n')
receipt={'read_at':datetime.now(timezone.utc).isoformat(),'report_reader_source':'808482fb50794cab506b3376d3729cee9105391d','run_output':run,'attempt_id':attempt_id,'model_trials_created':0,'evaluation_error':status.evaluation_error,'evaluations':len(status.evaluations),'failed_facts':summary.failed_facts,'summary_status':summary.status,'summary_output':str(out/'refreshed-attempt-summary.json')}
(out/'report-refresh-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
assert status.evaluation_error is None,status.evaluation_error
'''
compile(program,'s0-report-refresh','exec')
outer='import subprocess\nfrom marin.rl.skyrl import skyrl_cpu_command\nsubprocess.run(skyrl_cpu_command("python","-c",'+repr(program)+'),check=True)'
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-report-refresh-20261126b','-e','MARIN_PREFIX','s3://marin-us-east-08a/marin','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(args).returncode)
