"""Jerboa authorized ONE replacement per eight no-score startup failures, not best-of."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');N=P.parent/'t310/current-orders.md';C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing');ALLOWED={('A0',4),('A2',3),('A3',2),('A3',3),('A3',4),('A4',2),('A4',3),('A4',4)}
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--extra-index',type=int,required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();assert (a.arm,a.extra_index) in ALLOWED;os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 failed_dir=P/'bfcl-repeat20-r1'/a.arm/f'extra-{a.extra_index}';failed_raw=(failed_dir/'launch.json').read_bytes();failed=json.loads(failed_raw);error_raw=(failed_dir/'startup-error-record.actual.json').read_bytes();error=json.loads(error_raw);assert error['status']=='infra_failed' and error['metrics']=={} and not (failed_dir/'paired-score.json').exists();done=json.loads((failed_dir/'completion.json').read_bytes());assert done['SDK_terminal'] and done['state']=='failed'
 plan_raw=(P/'bfcl-ablation-r1/launch-plan.json').read_bytes();assert sha(plan_raw)=='0a85b86c3a910e4cea863df06cc1aa2627e037ac632167ed73ab406898978222';binding=next(x['bfcl'] for x in json.loads(plan_raw)['arms'] if x['arm']==a.arm);original=json.loads((P/'bfcl-ablation-r1'/a.arm/'bfcl-launch.json').read_bytes());model=Path(binding['model_config']);assert sha(model.read_bytes())==original['model_yaml_sha256']==failed['same_original_model_yaml_sha256'];assert sha(Path(binding['policy_wrapper']).read_bytes())==binding['policy_wrapper_sha256'];hf=original['hf_export'];D=P/'bfcl-startup-replacements-r1'/a.arm/f'extra-{a.extra_index}';OUT=D/'launch.json';assert not OUT.exists(),'Consumed one-replacement journal; NOsecondtry';load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();markers={}
 for name,key in [('path-b-checkpoint-event.json','event_sha256'),('path-b-storage-audit.json','audit_sha256')]:
  f,k=filesystem_for(hf+'/'+name);b=f.cat_file(k);assert sha(b)==original[key];markers[name]=sha(b)
 f,k=filesystem_for(failed['records_prefix']);objects=f.find(k);assert len(objects)==1 and objects[0].endswith('/record.json');assert sha(f.cat_file(objects[0]))==sha(error_raw)
 artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-startup-replacements-20261030-r1/{a.arm}/extra-{a.extra_index}';records=artifact+'/records';f,k=filesystem_for(records);assert not f.exists(k) and not f.find(k);orders=N.read_bytes();assert sha(orders)==a.orders_sha256;D.mkdir(parents=True,exist_ok=True);argv=list(binding['argv']);argv[argv.index('--model-config')+1]=str(model);version=f'pathb-bfcl-{a.arm.lower()}-extra{a.extra_index}-startup-replacement1';argv[argv.index('--version')+1]=version;argv[argv.index('--records-prefix')+1]=records;argv[argv.index('--description')+1]='Jerboa ONEstartup-no-score replacement '+a.arm+f' extra{a.extra_index};failedattemptretained';os.environ['MARIN_PREFIX']=artifact+'/native';os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 record={'state':'PRE_SUBMIT','utc':utc(),'authority':'JerboaONEreplacementeachof8startupfailures;notbestof;interactiveunchanged;>2newpreemptionsESCALATEbeforeotherwork','arm':a.arm,'additional_repeat_index':a.extra_index,'startup_replacement_number':1,'replaces_failed_job':failed['job'],'failed_launch_path':str(failed_dir/'launch.json'),'failed_launch_sha256':sha(failed_raw),'startup_error_record_sha256':sha(error_raw),'failed_native_status':error['status'],'failed_scores_not_zero_filled':True,'same_existing_HF':hf,'same_original_model_yaml':str(model),'same_original_model_yaml_sha256':sha(model.read_bytes()),'markers':markers,'records_prefix':records,'argv':argv,'method_pin':'caba0fa38b780af42da54af2f188ba5bfcc28501','no_sampling_seed_priority_changes':True,'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'no_automatic_retry_or_second_replacement':True,'native_checkpoint_ops':0,'training_submits':0}
 def save():
  with OUT.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with OUT.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 original_submit=IrisClient.submit
 def submit(client,*pos,**kwargs):
  assert record['application_submit_calls']==0 and N.read_bytes()==orders;kwargs['timeout']=Duration.from_hours(49);assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;expected='/karan/'+kwargs['name'];assert not client.list_jobs(prefix=expected);record.update(state='SUBMISSION_CALLED',expected_job=expected,application_submit_calls=1,called_utc=utc());save()
  try:job=original_submit(client,*pos,**kwargs)
  except Exception as e:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Readonly reconcile; no secondreplacement') from None
  record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();return job
 IrisClient.submit=submit
 try:
  from experiments.evaluation import held_bfcl
  sys.argv=[argv[2],*argv[3:]];held_bfcl.main()
 finally:IrisClient.submit=original_submit
 assert record['state']=='SUBMITTED';log=(D/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),'job':record['job'],'replaces_failed_job':failed['job']}),flush=True)
if __name__=='__main__':main()
