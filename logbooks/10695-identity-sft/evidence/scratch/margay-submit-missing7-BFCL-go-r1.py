"""Jerboa seven missing BFCL slots GO, plus at most one confirmed infra retry per slot."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');N=P.parent/'t310/current-orders.md';C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing');SLOTS={('A0',4),('A2',3),('A3',2),('A3',3),('A3',4),('A4',2),('A4',4)}
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--slot',type=int,required=True);ap.add_argument('--attempt',type=int,choices=[1,2],default=1);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();assert (a.arm,a.slot) in SLOTS;os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501';index=json.loads((P/'bfcl-repeat20-r1/all34-physical-and26-logical-run-index-margay-r3.json').read_bytes());slotid=a.arm+f'/extra-{a.slot}';logical=next(r for r in index['primary_runs'] if r['logical_slot']==slotid);assert logical['state']=='INFRA_FAILED_NO_SCORE'
 prior_dir=Path(logical['launch']).parent if a.attempt==1 else P/'bfcl-missing7-go-r1'/a.arm/f'extra-{a.slot}'/'attempt-1';prior=json.loads((prior_dir/'launch.json').read_bytes());terminal=json.loads((prior_dir/'completion.json').read_bytes());assert terminal['SDK_terminal'] and terminal['state']!='succeeded';error_path=prior_dir/'startup-error-record.actual.json';assert error_path.exists(),'Beforeoneinfraretry,inspectactualfailureandpreserverecord';error=json.loads(error_path.read_bytes());assert error['status']=='infra_failed' and error['metrics']=={};assert not (prior_dir/'paired-doc-reward-arrays.json').exists()
 raw=(P/'bfcl-ablation-r1/launch-plan.json').read_bytes();assert sha(raw)=='0a85b86c3a910e4cea863df06cc1aa2627e037ac632167ed73ab406898978222';binding=next(x['bfcl'] for x in json.loads(raw)['arms'] if x['arm']==a.arm);original=json.loads((P/'bfcl-ablation-r1'/a.arm/'bfcl-launch.json').read_bytes());model=Path(binding['model_config']);assert sha(model.read_bytes())==original['model_yaml_sha256']==logical['model_config_sha256'];assert sha(Path(binding['policy_wrapper']).read_bytes())==binding['policy_wrapper_sha256'];D=P/'bfcl-missing7-go-r1'/a.arm/f'extra-{a.slot}'/f'attempt-{a.attempt}';OUT=D/'launch.json';assert not OUT.exists(),'Consumedattemptjournal;readonlyreconcile';load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();hf=original['hf_export'];markers={}
 for name,key in [('path-b-checkpoint-event.json','event_sha256'),('path-b-storage-audit.json','audit_sha256')]:
  f,k=filesystem_for(hf+'/'+name);b=f.cat_file(k);assert sha(b)==original[key];markers[name]=sha(b)
 f,k=filesystem_for(prior['records_prefix']);objects=f.find(k);assert len(objects)==1 and objects[0].endswith('/record.json') and sha(f.cat_file(objects[0]))==sha(error_path.read_bytes())
 artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-missing7-go-20261030-r1/{a.arm}/extra-{a.slot}/attempt-{a.attempt}';records=artifact+'/records';f,k=filesystem_for(records);assert not f.exists(k) and not f.find(k);orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'BFCL repeats GO at INTERACTIVE' in orders;D.mkdir(parents=True,exist_ok=True);argv=list(binding['argv']);argv[argv.index('--model-config')+1]=str(model);argv[argv.index('--version')+1]=f'pathb-bfcl-{a.arm.lower()}-slot{a.slot}-go1-attempt{a.attempt}';argv[argv.index('--records-prefix')+1]=records;argv[argv.index('--description')+1]=f'Jerboa missing7 GO interactive {slotid} attempt{a.attempt}; retain infra failures';os.environ['MARIN_PREFIX']=artifact+'/native';os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 record={'state':'PRE_SUBMIT','opened_utc':utc(),'authority':'JerboaBFCLmissing7GOinteractive;atmostONEconfirmedinfraretryafterthisfreshattempt;noscoreselection;reportall7scoredorexhausted','arm':a.arm,'additional_repeat_index':a.slot,'logical_slot':slotid,'go_attempt':a.attempt,'max_application_attempts_this_order':2,'prior_failed_job':prior['job'],'prior_failed_launch':str(prior_dir/'launch.json'),'prior_error_record_sha256':sha(error_path.read_bytes()),'same_existing_HF':hf,'same_original_model_yaml':str(model),'same_original_model_yaml_sha256':sha(model.read_bytes()),'markers':markers,'method_pin':'caba0fa38b780af42da54af2f188ba5bfcc28501','records_prefix':records,'argv':argv,'priority':'interactive','sampling_temperature_seed_reward_parser_changes':False,'requested_native_failure_preemption_retries':[0,0],'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'native_checkpoint_ops':0,'training_submits':0,'no_automatic_retry_on_uncertain_submit':True}
 def save():
  with OUT.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with OUT.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 old=IrisClient.submit
 def submit(client,*pos,**kwargs):
  assert record['application_submit_calls']==0 and N.read_bytes()==orders;kwargs['timeout']=Duration.from_hours(49);assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;expected='/karan/'+kwargs['name'];assert not client.list_jobs(prefix=expected);record.update(state='SUBMISSION_CALLED',expected_job=expected,application_submit_calls=1,called_utc=utc());save()
  try:job=old(client,*pos,**kwargs)
  except Exception as e:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Readonlyreconcile;uncertainisnotinfraretrypermission') from None
  record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();return job
 IrisClient.submit=submit
 try:
  from experiments.evaluation import held_bfcl
  sys.argv=[argv[2],*argv[3:]];held_bfcl.main()
 finally:IrisClient.submit=old
 assert record['state']=='SUBMITTED';log=(D/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),'job':record['job'],'logical_slot':slotid,'go_attempt':a.attempt}),flush=True)
if __name__=='__main__':main()
