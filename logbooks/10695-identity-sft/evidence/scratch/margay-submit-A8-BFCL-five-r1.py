"""Ordered A8 HF-only BFCL originals; freeze original slot1 before diagnostic reads."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-a8-r1';N=P.parent/'t310/current-orders.md';C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing')
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A8a','A8b'],required=True);ap.add_argument('--readiness-sha256',required=True);ap.add_argument('--run',type=int,choices=range(1,6),required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 raw=(D/'launch-plan.json').read_bytes();assert sha(raw)=='f527fb76e294211e33e5e28eb1183bc6788ae5414f229542e047c410332e2a65';plan=json.loads(raw);target=next(t for t in plan['arms'] if t['arm']==a.arm);binding=target['bfcl'];assert sha(Path(binding['policy_wrapper']).read_bytes())==binding['policy_wrapper_sha256'];model=Path(binding['model_config']);assert sha(model.read_bytes())==binding['model_config_sha256']
 pp=D/a.arm/'actual-audited-HF-ready-margay-r1.json';assert sha(pp.read_bytes())==a.readiness_sha256;proof=json.loads(pp.read_bytes());assert proof['actual_bfcl_ready_marker'] and proof['all502_audit']=={'frozen_full_fields':104,'trained_master_cast':398,'frozen_names_equal_actual_training':True};load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3()
 for key in ('event.json','audit.json','native-retired.json','bfcl-ready.json'):
  item=proof['metadata'][key];fs,k=filesystem_for(item['uri']);assert sha(fs.cat_file(k))==item['sha256']
 sd=D/a.arm/'BFCL-five-r1'/f'run-{a.run}';out=sd/'launch.json';assert not out.exists(),'Consumedlogicalrun;NOautomaticretry';artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-a8-five-20261030-r1/{a.arm}/run-{a.run}';records=artifact+'/records';fs,k=filesystem_for(records);assert not fs.exists(k) and not fs.find(k);orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'ON-share dose arms A8a / A8b' in orders;argv=list(binding['argv']);argv[argv.index('--version')+1]=f'pathb-{a.arm.lower()}-u36-bfcl-five-r1-run{a.run}';argv[argv.index('--records-prefix')+1]=records;argv[argv.index('--description')+1]=f'Ordered{a.arm}HFonlyfull123BFCL {a.run}/5,actual502audit,unchangedmethod,noP1prefill';os.environ['MARIN_PREFIX']=artifact+'/native';os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None);sd.mkdir(parents=True,exist_ok=True)
 record={'state':'PRE_SUBMIT','opened_utc':utc(),'authority':'JerboaA8GO+Marten6fc95936BFCL5AFTERactualauditedHF;originalONEattempt;wholefreshreplacementonlyterminalincomplete','arm':a.arm,'evaluation_index':a.run,'model_yaml':str(model),'model_config_sha256':sha(model.read_bytes()),'HF':proof['HF'],'HF_readiness_proof':str(pp),'HF_readiness_proof_sha256':sha(pp.read_bytes()),'event_sha256':proof['metadata']['event.json']['sha256'],'audit_sha256':proof['metadata']['audit.json']['sha256'],'method_pin':'caba0fa38b780af42da54af2f188ba5bfcc28501','records_prefix':records,'argv':argv,'priority':'interactive','requested_native_retries':[0,0],'sampling_temperature_seed_reward_parser_changes':False,'no_P1_prefill':True,'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'native_checkpoint_ops':0,'replacement_number':0,'no_automatic_retry_on_uncertain_submission':True,'terminal_incomplete_original_one_whole_fresh_replacement_allowed':True,'primary_statistics_spec_sha256':'d6ba2832c39370a53674fd4bfb29b0dc3d9861558b945f69f0af6a11bb161f18','original_slot1_diagnostic_preregistration_sha256':'bbebfea8faa31d21cd5c570e453cf58c839a78f9d1d306ce87ab6109a5772d85','no_partial_primary_comparisons':True}
 def save():
  with out.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with out.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 old=IrisClient.submit
 def submit(client,*pos,**kwargs):
  assert record['application_submit_calls']==0 and N.read_bytes()==orders;kwargs['timeout']=Duration.from_hours(49);assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE;expected='/karan/'+kwargs['name'];assert not client.list_jobs(prefix=expected);record.update(state='SUBMISSION_CALLED',expected_job=expected,application_submit_calls=1,called_utc=utc());save()
  try:job=old(client,*pos,**kwargs)
  except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Readonlyreconcile;NOautomaticretry') from None
  record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();return job
 IrisClient.submit=submit
 try:
  from experiments.evaluation import held_bfcl
  sys.argv=[argv[2],*argv[3:]];held_bfcl.main()
 finally:IrisClient.submit=old
 assert record['state']=='SUBMITTED';log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save()
 if a.run==1:
  fp=P/'bfcl-failure-diagnostic-r1/diagnostic369-frozen-runs-docIDs-margay-r1.json';assert sha(fp.read_bytes())=='08020bffdd9b9c41222c446caededf5f705232178ac131045c7b7ad808e2b4e8';ids=json.loads(fp.read_bytes())['runs'][0]['doc_ids'];assert len(ids)==len(set(ids))==123;frozen={'arm':a.arm,'original_evaluation_slot':1,'replacement_number':0,'actual_original_job_id':record['job'],'launch':str(out),'launch_sha256':sha(out.read_bytes()),'doc_ids':ids,'preregistration_sha256':'bbebfea8faa31d21cd5c570e453cf58c839a78f9d1d306ce87ab6109a5772d85','frozen_before_any_session_read':True,'all123_including_passes':True,'native_eval_run_ID_must_join_THIS_original_job_before_session_read':True,'missing_original_explicit_no_first_success_or_replacement_substitution':True};ff=D/a.arm/'original-BFCL-slot1-diagnostic-cohort-margay-r1.json'
  with ff.open('x') as h:json.dump(frozen,h,indent=2);h.write('\n')
  print('FROZEN_ORIGINAL_SLOT1',str(ff),sha(ff.read_bytes()),flush=True)
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'run':a.run}),flush=True)
if __name__=='__main__':main()
