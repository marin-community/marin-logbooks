"""Exactly four additional same-method BFCL runs per audited A0-A4 export."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');N=P.parent/'t310/current-orders.md';C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing')
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A0','A1','A2','A3','A4'],required=True);ap.add_argument('--extra-index',type=int,choices=[1,2,3,4],required=True);ap.add_argument('--orders-sha256',required=True);args=ap.parse_args();os.chdir(C)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 plan_raw=(P/'bfcl-ablation-r1/launch-plan.json').read_bytes();assert sha(plan_raw)=='0a85b86c3a910e4cea863df06cc1aa2627e037ac632167ed73ab406898978222';arm=next(a for a in json.loads(plan_raw)['arms'] if a['arm']==args.arm);binding=arm['bfcl'];original_dir=P/'bfcl-ablation-r1'/args.arm;original_path=original_dir/'bfcl-launch.json';original_raw=original_path.read_bytes();original=json.loads(original_raw);model=Path(binding['model_config']);assert sha(model.read_bytes())==original['model_yaml_sha256'];assert sha(Path(binding['policy_wrapper']).read_bytes())==binding['policy_wrapper_sha256'];hf=original['hf_export'];D=P/'bfcl-repeat20-r1'/args.arm/f'extra-{args.extra_index}';OUT=D/'launch.json';assert not OUT.exists(),'Consumed repeat journal; no resubmit'
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();blobs={}
 for name,hkey in [('path-b-checkpoint-event.json','event_sha256'),('path-b-storage-audit.json','audit_sha256')]:
  f,k=filesystem_for(hf+'/'+name);b=f.cat_file(k);assert sha(b)==original[hkey];blobs[name]=b
 event=json.loads(blobs['path-b-checkpoint-event.json']);audit=json.loads(blobs['path-b-storage-audit.json']);counts=(26,476) if args.arm=='A4' else (104,398);assert event['hf_export']==hf and event['completed_updates']==36 and (audit['frozen_native_tensors'],audit['trained_master_cast_tensors'])==counts and len(audit['tensors'])==502
 artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-repeat20-20261030-r1/{args.arm}/extra-{args.extra_index}';records=artifact+'/records';f,k=filesystem_for(records);assert not f.exists(k) and not f.find(k);orders=N.read_bytes();assert sha(orders)==args.orders_sha256;D.mkdir(parents=True,exist_ok=True);argv=list(binding['argv']);argv[argv.index('--model-config')+1]=str(model);version=f'pathb-bfcl-{args.arm.lower()}-extra{args.extra_index}-20261030-r1';argv[argv.index('--version')+1]=version;argv[argv.index('--records-prefix')+1]=records;argv[argv.index('--description')+1]='Jerboa 4 additional same-method BFCL runs per arm; '+args.arm+f' extra{args.extra_index}';os.environ['MARIN_PREFIX']=artifact+'/native';os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 record={'state':'PRE_SUBMIT','utc':utc(),'authority':'Jerboa20additionalruns:fourperA0-A4,alloutcomes;A0total6others5;notretries;runmeans/bootstrapprimary','arm':args.arm,'additional_repeat_index':args.extra_index,'planned_total_runs_for_arm':6 if args.arm=='A0' else 5,'same_existing_HF':hf,'same_original_model_yaml':str(model),'same_original_model_yaml_sha256':sha(model.read_bytes()),'original_launch_receipt':str(original_path),'original_launch_receipt_sha256':sha(original_raw),'original_event_sha256':sha(blobs['path-b-checkpoint-event.json']),'original_audit_sha256':sha(blobs['path-b-storage-audit.json']),'records_prefix':records,'version':version,'scientific_binding_pin':'caba0fa38b780af42da54af2f188ba5bfcc28501','argv':argv,'sampling_or_seed_changes':False,'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'training_submit_calls':0,'native_reads_leases_deletes':0,'no_automatic_retry':True,'no_best_of_or_original_replacement':True,'outer_CPU_seconds':176400,'serve_seconds':43200,'cost_estimate_not_guarantee_H100h':1.3}
 def save():
  with OUT.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with OUT.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 original_submit=IrisClient.submit
 def submit(client,*pos,**kwargs):
  assert record['application_submit_calls']==0 and N.read_bytes()==orders;kwargs['timeout']=Duration.from_hours(49);assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;expected='/karan/'+kwargs['name'];assert not client.list_jobs(prefix=expected);record.update(state='SUBMISSION_CALLED',expected_job=expected,application_submit_calls=1,called_utc=utc());save()
  try:job=original_submit(client,*pos,**kwargs)
  except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile readonly, no repeat/retry') from None
  record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();return job
 IrisClient.submit=submit
 try:
  from experiments.evaluation import held_bfcl
  sys.argv=[argv[2],*argv[3:]];held_bfcl.main()
 finally:IrisClient.submit=original_submit
 assert record['state']=='SUBMITTED';log=(D/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),'arm':args.arm,'extra_index':args.extra_index,'job':record['job']}),flush=True)
if __name__=='__main__':main()
