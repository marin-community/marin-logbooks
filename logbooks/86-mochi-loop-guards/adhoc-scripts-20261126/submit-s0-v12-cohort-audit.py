import argparse
import json
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('cohort',choices=('tb2','tblite','swe100','swesmith-ho'))
parser.add_argument('--submit',action='store_true')
args=parser.parse_args()
state=json.loads(Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261126_0050_s0_v12_monitoring_state.json').read_text())
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12/partial/'+args.cohort
command=['-m','experiments.post_training.mochi_loop_guards.audit','--run-output',state['output'],'--output',output]
for cohort in ('tb2','tblite','swe100','swesmith-ho'):
 if cohort!=args.cohort:command.extend(('--exclude-cohort',cohort))
program='import subprocess,sys\nsubprocess.run([sys.executable,'+','.join(repr(x) for x in command)+'],check=True)'
compile(program,'s0-v12-cohort-audit','exec')
if not args.submit:
 print(json.dumps({'prepared':True,'cohort':args.cohort,'run_output':state['output'],'output':output,'model_trials':0}));raise SystemExit(0)
cli=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','16','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','7200','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-v12-audit-'+args.cohort+'-20261126a','-e','MARIN_PREFIX','s3://marin-us-east-08a/marin','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(cli).returncode)
