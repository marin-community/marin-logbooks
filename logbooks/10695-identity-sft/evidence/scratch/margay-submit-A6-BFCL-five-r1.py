"""Five ordered A6 HF-only BFCL evaluations after actual502 audit; one attempt each."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-a6-r1';N=P.parent/'t310/current-orders.md';C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing')
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=int,choices=range(1,6),required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 raw=(D/'launch-plan.json').read_bytes();assert sha(raw)=='3ffbca1c6adc01d54e990d4dada1d268bb93fb342394878b627ba7f63a8b7160';plan=json.loads(raw);binding=plan['arms'][0]['bfcl'];assert sha(Path(binding['policy_wrapper']).read_bytes())==binding['policy_wrapper_sha256'];model=Path(binding['model_config']);assert sha(model.read_bytes())==binding['model_config_sha256']=='4d29d3f9f515008f0d0f40666a451b69814e257cd06f80095bb2b051458dae31'
 pp=D/'actual-terminal-HF502-audit-retirement-ready-margay-r1.json';assert sha(pp.read_bytes())=='000610739f5630b79ee649ee8f07c2af35cc7143d74370a5b3b75e19a7c23a4d';proof=json.loads(pp.read_bytes());assert proof['actual_bfcl_ready_marker'] and proof['all502_audit']=={'frozen_full_fields':104,'trained_master_cast':398,'frozen_names_equal_actual_training':True};load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3()
 for key in ('event.json','audit.json','native-retired.json','bfcl-ready.json'):
  item=proof['metadata'][key];fs,k=filesystem_for(item['uri']);assert sha(fs.cat_file(k))==item['sha256']
 sd=D/'BFCL-five-r1'/f'run-{a.run}';out=sd/'launch.json';assert not out.exists(),'Consumedlogicalrun;NOautomaticretry';artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-a6-five-20261030-r1/run-{a.run}';records=artifact+'/records';fs,k=filesystem_for(records);assert not fs.exists(k) and not fs.find(k);orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'P2, A6 training arm' in orders;argv=list(binding['argv']);argv[argv.index('--version')+1]=f'pathb-a6-u36-bfcl-five-r1-run{a.run}';argv[argv.index('--records-prefix')+1]=records;argv[argv.index('--description')+1]=f'OrderedA6HFonlyfull123BFCL {a.run}/5,actual502audit,unchangedmethod,noP1prefill';os.environ['MARIN_PREFIX']=artifact+'/native';os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None);sd.mkdir(parents=True,exist_ok=True)
 record={'state':'PRE_SUBMIT','opened_utc':utc(),'authority':'JerboaP2+Marten41828orderedBFCL5AFTERactualauditedHF;ONEattemptperrun/noautomaticretry','arm':'A6','evaluation_index':a.run,'model_yaml':str(model),'model_config_sha256':sha(model.read_bytes()),'HF':proof['HF'],'HF_readiness_proof':str(pp),'HF_readiness_proof_sha256':sha(pp.read_bytes()),'event_sha256':proof['metadata']['event.json']['sha256'],'audit_sha256':proof['metadata']['audit.json']['sha256'],'method_pin':'caba0fa38b780af42da54af2f188ba5bfcc28501','records_prefix':records,'argv':argv,'priority':'interactive','requested_native_retries':[0,0],'sampling_temperature_seed_reward_parser_changes':False,'no_P1_prefill':True,'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'native_checkpoint_ops':0,'no_automatic_retry_on_uncertain_or_incomplete_run':True,'primary_statistics_spec_sha256':'c2cd0734debd3e6a32274e57a5fae0fceda4dd8e1e120f98313351a4cfec67b2','no_partial_primary_comparisons':True}
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
 assert record['state']=='SUBMITTED';log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'run':a.run}),flush=True)
if __name__=='__main__':main()
