import subprocess
program=r'''
import gzip,io,json,time,zipfile
from datetime import datetime,timezone
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from vendor.skyrl.scripts.loop_audit import trial_turns
import yaml
configure_coreweave_s3()
root=StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-8d1a0b19d0-in-af5ab9b8/2026.11.25.9')
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/s0-v12-first-pi-traces.json'
started=time.monotonic();last_log=-570;seen=set();records=[]
while time.monotonic()-started<10800:
 path=root/'resolved-launch.yaml'
 if path.exists():
  config=yaml.safe_load(path.read_text())['config']
  paths=(StoragePath(config['artifacts']['attempts_root'])/'trajectories/schema_v7/archives/phase=eval/step=eval_only/*.zip').glob()
  for archive_path in sorted(paths,key=str):
   if str(archive_path) in seen:continue
   seen.add(str(archive_path))
   with zipfile.ZipFile(io.BytesIO(archive_path.read_bytes())) as archive:
    for name in archive.namelist():
     if not name.startswith('records/') or not name.endswith('.json.gz'):continue
     record=json.loads(gzip.decompress(archive.read(name)))
     if 'long-loop-smartie-swesmith' not in (record['trajectory']['environment_extras'].get('data_source') or ''):continue
     turns=trial_turns(record['response'].get('messages') or [])
     if len(turns)<2:continue
     records.append({'record_id':record['record_id'],'instance_id':record['trajectory']['instance_id'],'repetition':record['trajectory']['repetition_id'],'data_source':record['trajectory']['environment_extras'].get('data_source'),'assistant_turns':len(turns),'nonempty_observation_batches':sum(bool(t.observation) for t in turns),'verification_status':(record.get('verification_result') or {}).get('status'),'archive':str(archive_path)})
     if len(records)>=5:break
   if len(records)>=5:break
 if len(records)>=5:break
 elapsed=time.monotonic()-started
 if elapsed-last_log>=570:
  print(json.dumps({'phase':'waiting_for_first_five_multiturn_traces','elapsed':round(elapsed),'available':len(records)}),flush=True);last_log=elapsed
 time.sleep(30)
report={'observed_at':datetime.now(timezone.utc).isoformat(),'run_output':str(root),'model_trials_created':0,'sample_n':len(records),'sample_with_observations':sum(r['nonempty_observation_batches']>0 for r in records),'records':records,'scientific_result':False,'purpose':'Verify actual Pi exact-restart feedback retention; not a baseline or intervention gate'}
StoragePath(output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
if len(records)<5:raise RuntimeError('Initial-trace diagnostic timed out before five retained multi-turn records')
if not report['sample_with_observations']:raise RuntimeError('No actual observations in the first five multi-turn traces')
'''
outer='from marin.rl.skyrl import skyrl_cpu_command\nimport subprocess\nsubprocess.run(skyrl_cpu_command("python","-c",'+repr(program)+'),check=True)'
compile(program,'first-traces','exec');compile(outer,'first-traces-driver','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','4','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','14400','--sync-package','marin-core','--job-name','mochi-loop-guards-v12-first-pi-traces-20261126a','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(args).returncode)
