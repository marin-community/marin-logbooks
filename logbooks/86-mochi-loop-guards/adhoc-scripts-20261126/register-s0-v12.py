import fcntl,json,os,re,tempfile
from pathlib import Path
from datetime import datetime,timedelta
folder=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards')
log=Path('/tmp/mochi-loop-guards/s0-v12-submit.log').read_text()
job=re.search(r'^coordinator_job_id: (\S+)',log,re.M)[1]
submitted=re.search(r"^submitted_at: '([^']+)'",log,re.M)[1]
output=re.search(r'^  output: (s3://\S+)',log,re.M)[1]
state={'stage':'S0v12','coordinator_job_id':job,'cluster':'marin','target_cluster':'cw-us-east-08a','submission_process':82883,'source_commit':'d0801e23e23c848683bc753a5b1a29c6296948fa','source_tree_hash':re.search('s0-baseline-([a-f0-9]+)-in',output)[1],'submitted_at':submitted,'next_probe_at':(datetime.fromisoformat(submitted)+timedelta(seconds=120)).isoformat(),'output':output,'artifact_revision':output.rsplit('/',1)[1],'restart_count':10,'resubmit_command':'source ~/.config/marin/credentials/cw-s3.env with export enabled; uv run --no-sync python /tmp/mochi-loop-guards/run-s0-v12.py','gpus':128,'trials':823,'priority':'production','all_racks_eligible':True,'tensor_parallel':1,'model_dtype':'bfloat16','cuda_graphs':'PIECEWISE','owner':'01a213ab-bdeb-7810-8fc7-66c9c4b174a8','preflight':'/tmp/mochi-loop-guards/s0-v12-preflight-receipt.json','supersedes':'v9/v10 traces invalid; v11 normalization-only repair superseded before scientific use; v12 covers normalized and exact replay','latest_signal':'Accepted; placement pending'}
p=folder/'scratch/20261126_0050_s0_v12_monitoring_state.json';p.write_text(json.dumps(state,indent=2)+'\n')
registry=folder.parent/'job-watchdog/registry.yaml'
block=f'''  - id: mochi-loop-guards-s0-v12
    owner: 01a213ab-bdeb-7810-8fc7-66c9c4b174a8
    description: Corrected S0 observation retention;823 trials,128GB200,production,all racks
    expected_terminal: SUCCEEDED
    stall_threshold_minutes: 240
    iris: {{cluster: marin, job: {job}}}
    probe:
      type: log_regex
      cluster: marin
      job: {job}
'''
with registry.with_suffix('.yaml.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);text=registry.read_text();assert 'id: mochi-loop-guards-s0-v12' not in text
 assert '\nentries:\n' in text
 text=text.replace('\nentries:\n','\nentries:\n'+block,1)
 with tempfile.NamedTemporaryFile('w',dir=registry.parent,prefix='.mochi-',delete=False) as temp:
  temp.write(text);tmp=Path(temp.name)
 os.chmod(tmp,registry.stat().st_mode&0o777);os.replace(tmp,registry)
print(json.dumps({'job_id':job,'output':output,'next_probe_at':state['next_probe_at'],'artifact_revision':state['artifact_revision']}))
