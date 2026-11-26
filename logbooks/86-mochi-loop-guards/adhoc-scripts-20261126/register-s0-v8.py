import fcntl
import json
import re
from datetime import datetime,timedelta
from pathlib import Path

folder=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards')
log=Path('/tmp/mochi-loop-guards/s0-v8-submit.log').read_text()
root=re.search(r'^coordinator_job_id: (\S+)',log,re.M)[1]
submitted=re.search(r"^submitted_at: '([^']+)'",log,re.M)[1]
output=re.search(r'^  output: (s3://\S+)',log,re.M)[1]
state={
 'stage':'S0v8','coordinator_job_id':root,'cluster':'marin','execution_cluster':'cw-us-east-08a',
 'submit_session_id':33106,'source_commit':'a56109c38e2f331ef2f659a8e70e116780ed6ef7','source_tree_hash':'4d578e566f',
 'source_dirty':False,'artifact':'s0-baseline-4d578e566f-in-af5ab9b8@2026.11.25.8','output':output,
 'submitted_at':submitted,'next_probe_at':(datetime.fromisoformat(submitted)+timedelta(seconds=120)).isoformat(),
 'restart_count':7,'resubmit_command':'uv run --no-sync python /tmp/mochi-loop-guards/run-s0.py',
 'excluded_nvlink_domains':[],'recovery':'Spread CPU rollout workers across existing hosts; allocation and scientific SkyRL config equal to v7 except placement',
 'preflight_receipt':'/tmp/mochi-loop-guards/s0-v8-runtime-preflight-receipt.json',
 'latest_signal':'Coordinator accepted; first admission inspection pending; no placement/result claim'
}
path=folder/'scratch/20261125_1800_s0_v8_monitoring_state.json'
path.write_text(json.dumps(state,indent=2)+'\n')
registry=folder.parent/'job-watchdog/registry.yaml'
with registry.with_suffix('.yaml.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 s=registry.read_text()
 pattern=r'(?m)^  - id: mochi-loop-guards-s0-v7\n(?:(?!^  - id: ).*\n)*'
 m=re.search(pattern,s)
 assert m and '01a213ab-bdeb-7810-8fc7-66c9c4b174a8' in m[0]
 block=f'''  - id: mochi-loop-guards-s0-v8
    owner: 01a213ab-bdeb-7810-8fc7-66c9c4b174a8
    description: Mochi loop-guards S0 v8, spread CPU workers, 128 GPUs, all racks eligible
    expected_terminal: SUCCEEDED
    stall_threshold_minutes: 240
    iris: {{cluster: marin, job: {root}}}
    probe:
      type: log_regex
      cluster: marin
      job: {root}
'''
 registry.write_text(s[:m.start()]+block+s[m.end():])
with (folder/'STATUS.md').open('a') as f:
 f.write(f'\n18:00:26 EST: S0v8 accepted {root}; output {output}; source a56109c38e2f331ef2f659a8e70e116780ed6ef7/tree4d578e566f. Explicit spread placement fixes head-pinned CPUworkers while preserving128GPU/128GiB/allracks/production and exact scientific SkyRL config (compared to v7 with placement removed). NativeS0realplan passes;52focusedtests includingreal2nodeRayplacement andantithrash/schema pass; requiredlint/typecheck pass. NativeYAMLregeneration now records unchangedantithrashdefaults in canonicalYAML/sidecar, retaining I3schema. Firstprobe18:02:26EST. Watchdog ownentryv7->v8 atomically replaced underlock; otherentries preserved. No GPUchild placementclaimyet. T0 unchanged.\n')
print(json.dumps({'job_id':root,'state':str(path),'next_probe_at':state['next_probe_at']}))
