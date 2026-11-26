import fcntl,json,os,re,tempfile,yaml
from datetime import datetime,timedelta,timezone
from pathlib import Path
root=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards');owner='01a213ab-bdeb-7810-8fc7-66c9c4b174a8';arms={};entries=[]
for arm,process in [('stop',72140),('rp',80470)]:
 s=Path('/tmp/mochi-loop-guards/s1-'+arm+'-submit.log').read_text();job=re.search(r'^coordinator_job_id: (\S+)',s,re.M)[1];submitted=re.search(r"^submitted_at: '([^']+)'",s,re.M)[1];output=re.search(r'^  output: (s3://\S+)',s,re.M)[1];arms[arm]={'coordinator_job_id':job,'submitted_at':submitted,'output':output,'native_wait_process':process,'gpus':64,'trials':723,'latest_signal':'Accepted; placement pending','terminal':False}
 entries.append({'id':'mochi-loop-guards-s1-'+arm,'owner':owner,'description':'S1 '+arm+';723 frozen trials,64GB200,production,all racks','expected_terminal':'SUCCEEDED','stall_threshold_minutes':240,'iris':{'cluster':'marin','job':job},'probe':{'type':'log_regex','cluster':'marin','job':job}})
state={'stage':'S1','source_commit':'d0801e23e23c848683bc753a5b1a29c6296948fa','source_tree':'8d1a0b19d0','artifact_revision':'2026.11.25.9','arms':arms,'next_probe_at':(max(datetime.fromisoformat(a['submitted_at']) for a in arms.values())+timedelta(seconds=120)).isoformat(),'owner':owner,'cluster':'marin','target_cluster':'cw-us-east-08a','baseline_seal':'s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12/baseline-seal.json','baseline_audit':'s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12/audit.json','target_cohorts':['tb2','tblite','swesmith-ho'],'boundary_rule_comment':7190461858,'S2_start_cutoff':'2026-11-26T13:01:36+00:00','S2_update_cutoff':'2026-11-27T01:01:36+00:00','experiment_deadline':'2026-11-27T17:01:36+00:00','repeat_stall_precision':{'genuine':29,'n':30,'passed':True},'resubmit_command':'Only after authorized recovery: export existing CW credentials, run-stage.py <s1-stop|s1-rp>2026.11.25.9; never duplicate active roots'}
p=root/'scratch/20261126_0614_s1_monitoring_state.json';p.write_text(json.dumps(state,indent=2)+'\n');registry=root.parent/'job-watchdog/registry.yaml'
with registry.with_suffix('.yaml.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);s=registry.read_text();s,n=re.subn(r'(?m)^  - id: mochi-loop-guards-s0-v12-postprocess\n.*?(?=^  - id: |\Z)','',s,flags=re.S);assert n in (0,1)
 assert 'State: succeeded' in Path('/tmp/mochi-loop-guards/s0-v12-postprocess-terminal.log').read_text()
 for e in entries:assert 'id: '+e['id'] not in s
 block=''.join('  '+l+'\n' for l in yaml.safe_dump(entries,sort_keys=False).splitlines());s=s.replace('\nentries:\n','\nentries:\n'+block,1)
 with tempfile.NamedTemporaryFile('w',dir=registry.parent,prefix='.mochi-',delete=False) as f:f.write(s);tmp=Path(f.name)
 os.chmod(tmp,registry.stat().st_mode&0o777);os.replace(tmp,registry)
 doc=yaml.safe_load(s);doc['entries']=entries;Path('/tmp/mochi-loop-guards/watchdog-s1-subset.yaml').write_text(yaml.safe_dump(doc,sort_keys=False))
print(json.dumps({'arms':arms,'next_probe_at':state['next_probe_at']}))
