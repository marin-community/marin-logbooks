import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

state_path=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261125_1248_s0_v3_monitoring_state.json')
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
        Path('/tmp/mochi-loop-guards/s0-v3-'+kind+'-latest.log').write_text(payload)
        if done.returncode:
            print('PROBE_FAILED',kind,flush=True)
            raise SystemExit(1)
    health=next((line for line in output['state'].splitlines() if line.startswith('State:')),'state unavailable')
    markers=('Evaluation Progress:','Loading checkpoint shards','Starting training','Capturing CUDA graphs','Checkpoint loading took','generation','Submitted:')
    signals=[line.split(' | ')[-1] for line in output['logs'].splitlines() if any(marker in line for marker in markers) and len(line)<1200]
    now=datetime.now(timezone.utc)
    signal=health+'; '+(signals[-1][-500:] if signals else 'no new phase marker')
    state.update(latest_signal=signal,latest_observation_at=now.isoformat(),next_probe_at=(now+timedelta(seconds=570)).isoformat())
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    print(now.isoformat(),signal,flush=True)
    if 'State: running' not in output['state']:
        print('JOB_STATE_CHANGED',flush=True)
        break
    if 'Traceback (most recent call last)' in output['logs'] or 'OutOfMemoryError' in output['logs'] or 'CUDA error:' in output['logs']:
        print('ERROR_SIGNAL',flush=True)
        break
