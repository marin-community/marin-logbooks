import fcntl
import json
import os
import re
import tempfile
from datetime import datetime,timedelta
from pathlib import Path

folder=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards')
log=Path('/tmp/mochi-loop-guards/s0-v9-submit.log').read_text()
job=re.search(r'^coordinator_job_id: (\S+)',log,re.M)[1]
submitted=re.search(r"^submitted_at: '([^']+)'",log,re.M)[1]
output=re.search(r'^  output: (s3://\S+)',log,re.M)[1]
tree=re.search(r's0-baseline-([0-9a-f]+)-in-',output)[1]
state={'stage':'S0v9','coordinator_job_id':job,'cluster':'marin','execution_cluster':'cw-us-east-08a','submit_session_id':31949,'source_commit':'496dd1b5a876f5a4f9750115a79c4715e2897ee8','source_tree_hash':tree,'source_dirty':False,'artifact':output.split('/')[-2]+'@'+output.split('/')[-1],'output':output,'submitted_at':submitted,'next_probe_at':(datetime.fromisoformat(submitted)+timedelta(seconds=120)).isoformat(),'restart_count':8,'resubmit_command':'uv run --no-sync python /tmp/mochi-loop-guards/run-s0.py','excluded_nvlink_domains':[],'recovery':'Serialization-safe stdlib actor diagnostic; explicit spread CPU workers retained; scientific config and allocation equal to v8','preflight_receipt':'/tmp/mochi-loop-guards/s0-v9-runtime-preflight-receipt.json','latest_signal':'Coordinator accepted; admission pending; no trial/result claim'}
path=folder/'scratch/20261125_1835_s0_v9_monitoring_state.json';path.write_text(json.dumps(state,indent=2)+'\n')
registry=folder.parent/'job-watchdog/registry.yaml'
with registry.with_suffix('.yaml.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 text=registry.read_text()
 match=re.search(r'(?m)^  - id: mochi-loop-guards-s0-v8\n(?:(?!^  - id: ).*\n)*',text)
 assert match and '01a213ab-bdeb-7810-8fc7-66c9c4b174a8' in match[0]
 block=f'''  - id: mochi-loop-guards-s0-v9
    owner: 01a213ab-bdeb-7810-8fc7-66c9c4b174a8
    description: Mochi loop-guards S0 v9, spread CPU workers and safe actor logs, 128 GPUs, all racks
    expected_terminal: SUCCEEDED
    stall_threshold_minutes: 240
    iris: {{cluster: marin, job: {job}}}
    probe:
      type: log_regex
      cluster: marin
      job: {job}
'''
 with tempfile.NamedTemporaryFile('w',dir=registry.parent,prefix='.mochi-registry-',delete=False) as temp:
  temp.write(text[:match.start()]+block+text[match.end():]);temp_path=Path(temp.name)
 os.chmod(temp_path,registry.stat().st_mode & 0o777)
 os.replace(temp_path,registry)
p=folder/'STATUS.md';text=p.read_text()
current=f'Current: S0 v9 accepted 18:35:32 EST, coordinator `{job}`. GPU source `496dd1b5a876f5a4f9750115a79c4715e2897ee8`, tree `{tree}`. Scientific SkyRL config and allocation equal v8; full S0 823 trials. V8 failed before trials because the added Loguru actor log captured an append-only sink; stdlib logging fixes the reproduced real-actor serialization failure. Two-node placement regression, actual actor serialization with a runtime file sink, native S0 plan, lint/typecheck pass. First admission probe 18:37:32 EST; state `scratch/20261125_1835_s0_v9_monitoring_state.json`. Watchdog entry replaced v8->v9 under lock. No placement/result claim yet; T0 unchanged.'
text,count=re.subn(r'(?m)^Current:.*$',lambda _:current,text,count=1);assert count==1
text+=f'\n18:35:32 EST: S0v9 accepted {job}, output{output}. Source496dd1b5a876f5a4f9750115a79c4715e2897ee8/tree{tree}; scientificconfig+allocationexactlyequalv8,823S0trials/128GPU/128GiB/production/allracks/TP1BF16/graphs. Actorserializationregressionreproducedv8failurewithappendlogfilesink, passesafterstdlibloggingfix; real2nodeplacement/nativeplan/lint/typecheckpass. Submitwaiter31949; firstprobe18:37:32EST. V8terminalFAILED/nocancelneeded/nodatacollected. Ownwatchdogv8->v9 replacedunderlock,otherspreserved. OriginalT0 unchanged.\n'
p.write_text(text)
print(json.dumps({'job_id':job,'state':str(path),'next_probe_at':state['next_probe_at']}))
