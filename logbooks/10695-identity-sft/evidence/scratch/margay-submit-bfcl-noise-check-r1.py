"""Owner-authorized ONE repeat per existingHF; preserve originals, no retraining."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');N=P.parent;C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing')
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['r5-bbase10-u35','a0-u36'],required=True);ap.add_argument('--orders-sha256',required=True);args=ap.parse_args();os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501';science=(P/'full-screen-execution-r1/execution-packet.json').read_bytes();assert sha(science)=='13b058213e399e402d36a3f1c5612dc885b4345381939e36aaec351268deebb2';base=next(x for x in json.loads(science)['candidates'] if x['label']=='pathb-base25-u52-r1')['bfcl']
 if args.target=='r5-bbase10-u35':
  original_dir=P/'r5-checkpoint-bfcl-r1/pathb-bbase-10-r5-4bc3742fe772/updates-35';original_path=original_dir/'launch.json';model=original_dir/'model.yaml';binding=base;updates=35;expected_score=21
 else:
  original_dir=P/'bfcl-ablation-r1/A0';original_path=original_dir/'bfcl-launch.json';a=next(x for x in json.loads((P/'bfcl-ablation-r1/launch-plan.json').read_bytes())['arms'] if x['arm']=='A0');binding=a['bfcl'];model=Path(binding['model_config']);updates=36;expected_score=29
 original_raw=original_path.read_bytes();original=json.loads(original_raw);original_score_raw=(original_dir/'paired-score.json').read_bytes();score=next(iter(json.loads(original_score_raw).values()));assert score['n']==123 and round(score['mean_pct']*123/100)==expected_score;assert sha(model.read_bytes())==original['model_yaml_sha256'];assert sha(Path(binding['policy_wrapper']).read_bytes())==binding['policy_wrapper_sha256'];hf=original['hf_export'];D=P/'bfcl-noise-r1'/args.target;OUT=D/'launch.json';assert not OUT.exists(),'Consumed noise repeat journal; never resubmit';load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();blobs={}
 for name,hkey in [('path-b-checkpoint-event.json','event_sha256'),('path-b-storage-audit.json','audit_sha256')]:
  f,k=filesystem_for(hf+'/'+name);b=f.cat_file(k);assert sha(b)==original[hkey];blobs[name]=b
 event=json.loads(blobs['path-b-checkpoint-event.json']);audit=json.loads(blobs['path-b-storage-audit.json']);assert event['hf_export']==hf and event['completed_updates']==updates and audit['frozen_native_tensors']==104 and audit['trained_master_cast_tensors']==398 and len(audit['tensors'])==502;artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-noise-20261030-r1/{args.target}';records=artifact+'/records';f,k=filesystem_for(records);assert not f.exists(k) and not f.find(k);orders_path=N/'t310/current-orders.md';orders=orders_path.read_bytes();assert sha(orders)==args.orders_sha256;D.mkdir(parents=True,exist_ok=True);argv=list(binding['argv']);argv[argv.index('--model-config')+1]=str(model);version=f'pathb-bfcl-noise-{args.target}-repeat1-20261030';argv[argv.index('--version')+1]=version;argv[argv.index('--records-prefix')+1]=records;argv[argv.index('--description')+1]='Authorized ONE BFCL noise repeat '+args.target+' existing auditedHF; preserve original';os.environ['MARIN_PREFIX']=artifact+'/native';os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 record={'state':'PRE_SUBMIT','utc':utc(),'authority':'Jerboa explicitly ONErepeat existingr5Base10@35 plusONEcheapA0repeat; unchangedmethod/HF/YAML, freshnamespaces; no furtherrepeats/training','target':args.target,'same_existing_HF':hf,'same_original_model_yaml':str(model),'same_original_model_yaml_sha256':sha(model.read_bytes()),'original_launch_receipt':str(original_path),'original_launch_receipt_sha256':sha(original_raw),'original_score_sha256':sha(original_score_raw),'original_correct':expected_score,'original_n':123,'repeat_number':1,'original_event_sha256':sha(blobs['path-b-checkpoint-event.json']),'original_audit_sha256':sha(blobs['path-b-storage-audit.json']),'records_prefix':records,'version':version,'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'training_submit_calls':0,'native_reads_leases_deletes':0,'no_automatic_retry':True,'no_best_of_or_original_replacement':True,'cheap_A0_basis':'original A0BFCL measured1.308493333H100h; not a futurecostguarantee','outer_CPU_seconds':176400,'serve_seconds':43200}
 def save():
  with OUT.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with OUT.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 original_submit=IrisClient.submit
 def submit(client,*pos,**kwargs):
  assert record['application_submit_calls']==0 and orders_path.read_bytes()==orders;kwargs['timeout']=Duration.from_hours(49);assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;expected='/karan/'+kwargs['name'];assert not client.list_jobs(prefix=expected);record.update(state='SUBMISSION_CALLED',expected_job=expected,application_submit_calls=1,called_utc=utc());save()
  try:job=original_submit(client,*pos,**kwargs)
  except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile readonly, no repeat/retry') from None
  record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();return job
 IrisClient.submit=submit
 try:
  from experiments.evaluation import held_bfcl
  sys.argv=[argv[2],*argv[3:]];held_bfcl.main()
 finally:IrisClient.submit=original_submit
 assert record['state']=='SUBMITTED';log=(D/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),**record}),flush=True)
if __name__=='__main__':main()
