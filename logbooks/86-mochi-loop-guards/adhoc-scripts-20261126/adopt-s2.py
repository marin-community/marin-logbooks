import json,re,yaml,fcntl
from pathlib import Path
from datetime import datetime,timedelta,timezone
base=Path('/tmp/mochi-loop-guards');state={'stage':'S2_ANTITHRASH_PAIR','owner':'01a218ad-6f81-78f3-8e73-29b1684d79f6','source_commit':'0987bdac6f','source_tree':'6658f0dae1','artifact_revision':'2026.11.26.3','cluster':'marin','target_cluster':'cw-us-east-08a','N':4,'primary_measure':'agent_loop/repeat_turn_share','no_new_update_after':'2026-11-27T20:01:36-05:00','private_HF_blocker':'open-athena private-storage quota','restart_count':0,'arms':{}}
entries=[]
for arm,session in [('control',18357),('treatment',30309)]:
 text=(base/f's2-{arm}-submit-final.log').read_text();job=re.search(r'^coordinator_job_id: (\S+)',text,re.M).group(1);accepted=re.search(r"^submitted_at: '([^']+)'",text,re.M).group(1)
 output='s3://marin-us-east-08a/tmp/ttl=3d/marin/users/benfeuer/checkpoints/mochi-loop-guards/s2-antithrash-'+arm+'-6658f0dae1-in-db57e3bd/2026.11.26.3'
 state['arms'][arm]={'coordinator_job_id':job,'submitted_at':accepted,'output':output,'native_wait_process':session,'gpus':60,'N':4,'terminal':False,'latest_signal':'COORDINATOR_ACCEPTED; GPU child pending','latest_observation_at':accepted}
 entries.append({'id':'mochi-loop-guards-s2-'+arm,'owner':state['owner'],'description':'S2 antithrash '+arm+'; N4,60GB200,production08a; no new update after Nov27 20:01:36EST','expected_terminal':'SUCCEEDED','stall_threshold_minutes':180,'iris':{'cluster':'marin','job':job},'probe':{'type':'log_regex','cluster':'marin','job':job}})
state['next_probe_at']=(max(datetime.fromisoformat(x['submitted_at']) for x in state['arms'].values())+timedelta(seconds=120)).isoformat()
path=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261126_1143_s2_monitoring_state.json');path.write_text(json.dumps(state,indent=2)+'\n')
(base/'s2-submission-receipt.json').write_text(json.dumps(state,indent=2)+'\n')
registry=Path('/Users/benfeuer/Documents/experiments/active/job-watchdog/registry.yaml')
with registry.with_suffix('.yaml.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);d=yaml.safe_load(registry.read_text());ids={'mochi-loop-guards-s2-n4-preflight','mochi-loop-guards-conservation',*(r['id'] for r in entries)};d['entries']=[x for x in d['entries'] if x['id'] not in ids];d['entries'].extend(entries);registry.write_text(yaml.safe_dump(d,sort_keys=False))
(base/'watchdog-s2-subset.yaml').write_text(yaml.safe_dump({'coordinator':d['coordinator'],'defaults':d['defaults'],'entries':entries},sort_keys=False))
print(json.dumps({'jobs':{k:v['coordinator_job_id'] for k,v in state['arms'].items()},'next_probe_at':state['next_probe_at']}))
