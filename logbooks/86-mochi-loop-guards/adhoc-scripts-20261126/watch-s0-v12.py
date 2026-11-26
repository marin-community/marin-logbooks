import json
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from rigging.redaction import redact_string

state_path=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261126_0050_s0_v12_monitoring_state.json')
while True:
    state=json.loads(state_path.read_text())
    delay=max(0,datetime.fromisoformat(state['next_probe_at']).timestamp()-time.time())
    while delay>0:
        time.sleep(min(30,delay))
        delay=max(0,datetime.fromisoformat(state['next_probe_at']).timestamp()-time.time())
    job=state['coordinator_job_id']
    output={}
    for kind,args in [('state',['describe',job]),('logs',['logs',job,'--since-seconds','900','--max-lines','5000'])]:
        done=subprocess.run(['uv','run','--no-sync','iris','--cluster','marin','job',*args],capture_output=True,text=True,timeout=60)
        payload=(done.stdout+done.stderr).split('\nStepSpec(')[0]
        output[kind]=payload
        Path('/tmp/mochi-loop-guards/s0-v12-'+kind+'-latest.log').write_text(payload)
        if done.returncode:
            print('PROBE_FAILED',kind,flush=True)
            raise SystemExit(1)
    child_match=re.search(r'\[rl-iris\] Submitted: (\S+)',output['logs'])
    if child_match:
        state['gpu_job_id']=child_match.group(1)
    child_done=subprocess.run(['uv','run','--no-sync','iris','--cluster','marin','job','describe',state.get('gpu_job_id',job)],capture_output=True,text=True,timeout=60)
    output['child-state']=child_done.stdout+child_done.stderr
    Path('/tmp/mochi-loop-guards/s0-v12-child-state-latest.log').write_text(output['child-state'])
    if child_done.returncode:
        print('PROBE_FAILED child-state',flush=True)
        raise SystemExit(1)
    health=next((line for line in output['state'].splitlines() if line.startswith('State:')),'state unavailable')
    markers=('Evaluation Progress:','Loading checkpoint shards','Starting training','Capturing CUDA graphs','Checkpoint loading took','generation','Submitted:')
    signals=[line.split(' | ')[-1] for line in output['logs'].splitlines() if any(marker in line for marker in markers) and len(line)<1200]
    now=datetime.now(timezone.utc)
    child=next((line for line in output['child-state'].splitlines() if line.startswith('Tasks:')),'child state unavailable')
    signal=health+'; '+child+'; '+(signals[-1][-500:] if signals else 'no new phase marker')
    signal=redact_string(signal)
    persisted=json.loads(state_path.read_text())
    if state.get('gpu_job_id'):
        persisted['gpu_job_id']=state['gpu_job_id']
    state=persisted
    match=re.search(r'\[rl-iris\] Submitted: (\S+)',output['logs'])
    if match:
        state['gpu_job_id']=match.group(1)
    state.update(latest_signal=signal,latest_observation_at=now.isoformat(),next_probe_at=(now+timedelta(seconds=570)).isoformat())
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    print(now.isoformat(),signal,flush=True)
    if 'State: running' not in output['state']:
        print('JOB_STATE_CHANGED',flush=True)
        break
