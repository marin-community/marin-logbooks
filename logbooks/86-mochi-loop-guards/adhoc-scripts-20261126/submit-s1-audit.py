"""Audit a terminal follow-up, preserve its closed trace sink, refresh saved-data reporting."""
import argparse
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--arm',required=True,choices=('stop','rp','stop-repeat1','rp-repeat1'))
parser.add_argument('--run-output',required=True)
parser.add_argument('--coordinator-job',required=True)
parser.add_argument('--gpu-job',required=True)
parser.add_argument('--submit',action='store_true')
a=parser.parse_args()
assert a.run_output.startswith('s3://marin-us-east-08a/')
assert a.coordinator_job.startswith('/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s1-')
assert a.gpu_job.startswith(a.coordinator_job+'/')
baseline='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12/audit.json'
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s1-2026.11.26.1/'+a.arm
retained='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/traces/s1-'+a.arm
command=['-m','experiments.post_training.mochi_loop_guards.audit','--run-output',a.run_output,'--output',output,'--baseline',baseline,'--exclude-cohort','swe100']
report_program='\n'.join([
 'import json',
 'from marin.rl.skyrl_records import read_stored_skyrl_attempts',
 'from experiments.post_training import rl_compose',
 'from experiments.post_training.mochi_loop_guards.__main__ import GUARDS',
 'from experiments.post_training.rl_measurements import evaluation_status,read_recorded_launch',
 'from experiments.post_training.rl_publication import build_record',
 'from rigging.filesystem.s3_compat import configure_coreweave_s3',
 'from rigging.filesystem.storage_path import StoragePath',
 'configure_coreweave_s3()',
 "rl_compose.SELECTION_ROOT='s3://marin-us-east-08a/marin/rl-data/task-selection'",
 f'run={a.run_output!r};out=StoragePath({output!r})',
 'current,attempts=read_stored_skyrl_attempts(run);attempt=attempts[current]',
 f'preset=GUARDS.by_name[{("s1-stop" if a.arm.startswith("stop") else "s1-rp")!r}]',
 'status=evaluation_status(run,preset,attempt,read_recorded_launch(attempt))',
 '(out/"refreshed-evaluation-status.json").write_text(status.model_dump_json(indent=2)+"\\n")',
 'summary=build_record(run,preset,table=GUARDS,attempt=attempt)',
 '(out/"refreshed-attempt-summary.json").write_text(summary.model_dump_json(indent=2)+"\\n")',
 'assert status.evaluation_error is None,status.evaluation_error',
])
program='\n'.join([
 'import hashlib,json,subprocess,sys,yaml',
 'from marin.rl.skyrl import skyrl_cpu_command',
 'from rigging.filesystem.s3_compat import configure_coreweave_s3',
 'from rigging.filesystem.storage_path import StoragePath',
 'configure_coreweave_s3()',
 f'run_output={a.run_output!r};output={output!r}',
 'subprocess.run([sys.executable,'+','.join(repr(x) for x in command)+'],check=True)',
 'native=yaml.safe_load((StoragePath(run_output)/"resolved-launch.yaml").read_text())["config"]',
 'source=native["skyrl"]["generator"]["trajectory_retention"]["output_path"]',
 f'destination={retained!r}',
 'assert source.startswith("s3://marin-us-east-08a/")',
 'subprocess.run(["uv","run","--package","marin-core","fsutil","verified-copy",source,destination,"--workers","8"],check=True)',
 'manifest=(StoragePath(destination)/".verified-copy-manifest.json").read_bytes()',
 '(StoragePath(output)/"retention-promotion.json").write_text(json.dumps({"source":source,"destination":destination,"completion_manifest":destination+"/.verified-copy-manifest.json","manifest_sha256":hashlib.sha256(manifest).hexdigest(),"same_region":True},indent=2)+"\\n")',
 f'subprocess.run(skyrl_cpu_command("python","-c",{report_program!r}),check=True)',
 '(StoragePath(output)/"postprocessing-receipt.json").write_text(json.dumps({"complete":True,"model_trials_created":0,"run_output":run_output,"baseline":'+repr(baseline)+'},indent=2)+"\\n")',
])
compile(program,'s1-postprocess','exec');compile(report_program,'s1-report-refresh','exec')
if not a.submit:
 print(json.dumps({'prepared':True,'run_output':a.run_output,'baseline':baseline,'output':output,'retention_destination':retained,'model_trials_created':0,'excluded_cohorts':['swe100']}));raise SystemExit(0)
for kind,job in (('coordinator',a.coordinator_job),('gpu',a.gpu_job)):
 r=subprocess.run(['uv','run','--no-sync','iris','--cluster','marin','job','describe',job],capture_output=True,text=True,check=True)
 payload=r.stdout.split('\nStepSpec(')[0]
 status=next(line for line in payload.splitlines() if line.startswith('State:'))
 if status.split()[1] not in ('succeeded','failed','killed'):raise SystemExit('Wait for terminal '+kind+' before copying shared trace sink')
 Path('/tmp/mochi-loop-guards/s1-'+a.arm+'-pre-postprocess-'+kind+'.log').write_text(payload)
cli=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','16','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','14400','--sync-package','marin-core','--job-name','mochi-loop-guards-s1-postprocess-'+a.arm+'-20261126a','-e','MARIN_PREFIX','s3://marin-us-east-08a/marin','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(cli).returncode)
