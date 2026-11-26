import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('arm',choices=('stop','rp','rp-repeat1'));p.add_argument('cohort',choices=('tb2','tblite','swesmith-ho'));p.add_argument('--submit',action='store_true');a=p.parse_args()
state_name='20261126_1111_s1_rp_repeat1_monitoring_state.json' if a.arm=='rp-repeat1' else '20261126_0614_s1_monitoring_state.json';state=json.loads((Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch')/state_name).read_text());run=state['arms'][a.arm]['output'];baseline=state['baseline_audit'];output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s1-2026.11.26.1/'+a.arm+'/partial/'+a.cohort
command=['-m','experiments.post_training.mochi_loop_guards.audit','--run-output',run,'--output',output,'--baseline',baseline]
for c in ('tb2','tblite','swe100','swesmith-ho'):
 if c!=a.cohort:command.extend(('--exclude-cohort',c))
program='import subprocess,sys\nsubprocess.run([sys.executable,'+','.join(repr(x) for x in command)+'],check=True)';compile(program,'s1-cohort-audit','exec')
if not a.submit:print(json.dumps({'prepared':True,'arm':a.arm,'cohort':a.cohort,'run_output':run,'baseline':baseline,'output':output,'model_trials_created':0}));raise SystemExit(0)
cli=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','16','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','7200','--sync-package','marin-core','--job-name','mochi-loop-guards-s1-'+a.arm+'-audit-'+a.cohort+'-20261126a','-e','MARIN_PREFIX','s3://marin-us-east-08a/marin','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(cli).returncode)
