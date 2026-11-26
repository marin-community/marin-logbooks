import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

state_path=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261126_1143_s2_monitoring_state.json')
receipt_path=state_path.with_name('20261127_2001_s2_budget_cutoff_receipt.json')
state=json.loads(state_path.read_text())
while True:
 cutoff=datetime.fromisoformat(state['no_new_update_after']).timestamp()
 remaining=cutoff-time.time()
 if remaining>0:
  time.sleep(min(30,remaining))
  continue
 actions=[]
 for arm,item in state['arms'].items():
  job=item['coordinator_job_id']
  result=subprocess.run(['uv','run','--no-sync','iris','--cluster','marin','job','describe',job],capture_output=True,text=True,check=True)
  status=next(line for line in result.stdout.splitlines() if line.startswith('State:')).split()[1]
  if status in ('succeeded','failed','killed'):
   actions.append({'arm':arm,'job':job,'state':status,'action':'already terminal'})
   continue
  environment=dict(os.environ,IRIS_AGENT_PREFIX='/benfeuer/rl-data-task-health-attempts-by-run-users-benfeuer-checkpoints-mochi-loop-guards-s2-')
  result=subprocess.run([str(Path.home()/'.local/bin/iris-cancel-own'),'--cluster','marin',job],env=environment,capture_output=True,text=True,check=True)
  actions.append({'arm':arm,'job':job,'action':'canceled at registered T32 update cutoff'})
 receipt_path.write_text(json.dumps({'cutoff':state['no_new_update_after'],'authorization':'DESIGN no new update after T32; binding +24h user override','actions':actions},indent=2)+'\n')
 print(json.dumps({'receipt':str(receipt_path),'actions':actions}),flush=True)
 break
