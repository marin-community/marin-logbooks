import json,re,subprocess,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from rigging.redaction import redact_string
state_path=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261126_0614_s1_monitoring_state.json')
while True:
 state=json.loads(state_path.read_text());delay=max(0,datetime.fromisoformat(state['next_probe_at']).timestamp()-time.time())
 while delay>0:
  time.sleep(min(30,delay));delay=max(0,datetime.fromisoformat(state['next_probe_at']).timestamp()-time.time())
 updates={}
 for arm,item in state['arms'].items():
  if item.get('terminal'):continue
  output={}
  for kind,args in [('state',['describe',item['coordinator_job_id']]),('logs',['logs',item['coordinator_job_id'],'--since-seconds','900','--max-lines','5000'])]:
   done=subprocess.run(['uv','run','--no-sync','iris','--cluster','marin','job',*args],capture_output=True,text=True,timeout=60);payload=(done.stdout+done.stderr).split('\nStepSpec(')[0];output[kind]=payload;Path('/tmp/mochi-loop-guards/s1-'+arm+'-'+kind+'-latest.log').write_text(payload)
   if done.returncode:raise RuntimeError('Probe failed: '+arm+' '+kind)
  match=re.search(r'\[rl-iris\] Submitted: (\S+)',output['logs'])
  if match:item['gpu_job_id']=match.group(1)
  child='child submission pending'
  if item.get('gpu_job_id'):
   done=subprocess.run(['uv','run','--no-sync','iris','--cluster','marin','job','describe',item['gpu_job_id']],capture_output=True,text=True,timeout=60);payload=done.stdout+done.stderr;Path('/tmp/mochi-loop-guards/s1-'+arm+'-child-state-latest.log').write_text(payload)
   if done.returncode:raise RuntimeError('Child probe failed: '+arm)
   child=next((l for l in payload.splitlines() if l.startswith('Tasks:')),'child task state unavailable')
  health=next((l for l in output['state'].splitlines() if l.startswith('State:')),'state unavailable');lines=output['logs'].splitlines();progress=[l.split(' | ')[-1] for l in lines if 'Evaluation Progress:' in l and len(l)<1500];signals=progress or [l.split(' | ')[-1] for l in lines if any(k in l for k in ('Loading checkpoint shards','Capturing CUDA graphs','generation','Submitted:')) and len(l)<1500]
  now=datetime.now(timezone.utc);signal=redact_string(health+'; '+child+'; '+(signals[-1][-500:] if signals else 'no new phase marker'));item.update(latest_signal=signal,latest_observation_at=now.isoformat(),terminal=('State: running' not in output['state']));updates[arm]=item;print(now.isoformat(),arm,signal,flush=True)
 persisted=json.loads(state_path.read_text());persisted['arms'].update(updates);persisted['next_probe_at']=(datetime.now(timezone.utc)+timedelta(seconds=570)).isoformat();state_path.write_text(json.dumps(persisted,indent=2)+'\n')
 if all(a.get('terminal') for a in persisted['arms'].values()):print('ALL_S1_JOB_STATES_TERMINAL',flush=True);break
