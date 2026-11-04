"""Five FRESH v0.1 BFCL references; one attempt each, no historical34 pooling."""
import argparse, datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
D=P/'bfcl-v01-reference5-r1'; N=P.parent/'t310/current-orders.md'
C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing')
def sha(b): return hashlib.sha256(b).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run',type=int,choices=range(1,6),required=True); ap.add_argument('--orders-sha256',required=True); a=ap.parse_args()
 os.chdir(C); assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 raw=(P/'bfcl-ablation-r1/launch-plan.json').read_bytes(); assert sha(raw)=='0a85b86c3a910e4cea863df06cc1aa2627e037ac632167ed73ab406898978222'
 binding=next(x['bfcl'] for x in json.loads(raw)['arms'] if x['arm']=='A0'); assert sha(Path(binding['policy_wrapper']).read_bytes())==binding['policy_wrapper_sha256']
 model=D/'model-v01.yaml'; assert sha(model.read_bytes())=='d24f047a6c5c908b279a312d18cbb6f9d0969ac56b8f54c08e5d1400f1ca6730'
 assert model.read_bytes().splitlines(keepends=True)[2:]==(P/'bfcl-ablation-r1/model-A0.yaml').read_bytes().splitlines(keepends=True)[2:]
 proof=D/'plain-v01-binding-and-source-proof-r1.json'; assert sha(proof.read_bytes())=='ec36aa1bfd37b173a3612ecaba9e273ea7220c4990f748b3e5024f689d595bd5'; proof_data=json.loads(proof.read_bytes())
 for x in proof_data['source_bindings'].values(): assert sha(Path(x['local']).read_bytes())==x['sha256']
 sd=D/f'run-{a.run}'; out=sd/'launch.json'; assert not out.exists(),'Consumedjournal;readonlyreconcile'
 load_dotenv('/home/karan/oa/marin/.env'); configure_coreweave_s3()
 for x in proof_data['metadata'].values():
  fs,k=filesystem_for(x['uri']); assert sha(fs.cat_file(k))==x['sha256']
 artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-v01-reference5-20261030-r1/run-{a.run}'; records=artifact+'/records'; fs,k=filesystem_for(records); assert not fs.exists(k) and not fs.find(k)
 orders=N.read_bytes(); assert sha(orders)==a.orders_sha256 and b'v0.1 BFCL reference x5 NEW runs' in orders and b'old 34/123 is NOT pooled' in orders
 argv=list(binding['argv']); argv[argv.index('--model-config')+1]=str(model); argv[argv.index('--version')+1]=f'pathb-v01-reference5-r1-run{a.run}'; argv[argv.index('--records-prefix')+1]=records; argv[argv.index('--description')+1]=f'Jerboa FIVEFRESH plainv0.1 BFCL reference {a.run}/5; historical34 NOTpooled; unchangedmethod'
 os.environ['MARIN_PREFIX']=artifact+'/native'; os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None); sd.mkdir(parents=True,exist_ok=True)
 record={'state':'PRE_SUBMIT','opened_utc':utc(),'authority':'JerboaNEXT5NEWplainv01BFCLrefs;ONEattemptEACH;historical34NOTPOOLED','arm':'v0.1','reference_run_index':a.run,'max_application_attempts_this_order':1,'model_yaml':str(model),'model_yaml_sha256':sha(model.read_bytes()),'HF':proof_data['only_A0_config_deltas']['location']['new'],'source_proof':str(proof),'source_proof_sha256':sha(proof.read_bytes()),'method_pin':'caba0fa38b780af42da54af2f188ba5bfcc28501','records_prefix':records,'argv':argv,'priority':'interactive','sampling_temperature_seed_reward_parser_changes':False,'requested_native_failure_preemption_retries':[0,0],'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'native_checkpoint_ops':0,'training_submits':0,'no_automatic_retry_on_uncertain_submit':True,'historical34_pooled':False}
 def save():
  with out.open('w') as h: json.dump(record,h,indent=2); h.write('\n'); h.flush(); os.fsync(h.fileno())
 with out.open('x') as h: json.dump(record,h,indent=2); h.write('\n'); h.flush(); os.fsync(h.fileno())
 old=IrisClient.submit
 def submit(client,*pos,**kwargs):
  assert record['application_submit_calls']==0 and N.read_bytes()==orders; kwargs['timeout']=Duration.from_hours(49); assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0; assert kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE
  expected='/karan/'+kwargs['name']; assert not client.list_jobs(prefix=expected); record.update(state='SUBMISSION_CALLED',expected_job=expected,application_submit_calls=1,called_utc=utc()); save()
  try: job=old(client,*pos,**kwargs)
  except Exception as e:
   record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__); save(); raise RuntimeError('Readonlyreconcile;NOautomaticretry') from None
  record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc()); assert record['job']==expected; save(); return job
 IrisClient.submit=submit
 try:
  from experiments.evaluation import held_bfcl
  sys.argv=[argv[2],*argv[3:]]; held_bfcl.main()
 finally: IrisClient.submit=old
 assert record['state']=='SUBMITTED'; log=(sd/'completion.log').open('ab'); proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); log.close(); record['one_completion_notifier_pid']=proc.pid; save(); print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'reference_run_index':a.run}),flush=True)
if __name__=='__main__': main()
