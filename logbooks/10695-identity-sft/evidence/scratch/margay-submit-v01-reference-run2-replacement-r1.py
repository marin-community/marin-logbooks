"""Run2's ONE whole fresh replacement after its verified118+5infra terminal outcome."""
import argparse, datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
D=P/'bfcl-v01-reference5-r1'; R=P/'bfcl-v01-reference5-replacements-r1'; N=P.parent/'t310/current-orders.md'
C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing')
def sha(b): return hashlib.sha256(b).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run',type=int,choices=[2],required=True); ap.add_argument('--orders-sha256',required=True); ap.add_argument('--coverage-index',type=Path,required=True); a=ap.parse_args()
 os.chdir(C); assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 raw=(P/'bfcl-ablation-r1/launch-plan.json').read_bytes(); assert sha(raw)=='0a85b86c3a910e4cea863df06cc1aa2627e037ac632167ed73ab406898978222'
 binding=next(x['bfcl'] for x in json.loads(raw)['arms'] if x['arm']=='A0'); assert sha(Path(binding['policy_wrapper']).read_bytes())==binding['policy_wrapper_sha256']
 model=D/'model-v01.yaml'; assert sha(model.read_bytes())=='d24f047a6c5c908b279a312d18cbb6f9d0969ac56b8f54c08e5d1400f1ca6730'
 assert model.read_bytes().splitlines(keepends=True)[2:]==(P/'bfcl-ablation-r1/model-A0.yaml').read_bytes().splitlines(keepends=True)[2:]
 proof=D/'plain-v01-binding-and-source-proof-r1.json'; assert sha(proof.read_bytes())=='ec36aa1bfd37b173a3612ecaba9e273ea7220c4990f748b3e5024f689d595bd5'; proof_data=json.loads(proof.read_bytes())
 for x in proof_data['source_bindings'].values(): assert sha(Path(x['local']).read_bytes())==x['sha256']
 coverage=json.loads(a.coverage_index.read_bytes()); refs=[x for x in coverage['physical_reference_runs'] if x['replacement_number']==0]; assert len(refs)==5 and {x['reference_run_index'] for x in refs}==set(range(1,6))
 for x in refs:
  prior_lp=D/f'run-{x["reference_run_index"]}'/'launch.json'; prior_done=json.loads((prior_lp.parent/'completion.json').read_bytes()); assert x['run_id']==json.loads(prior_lp.read_bytes())['job'] and x['launch_sha256']==sha(prior_lp.read_bytes())
  if x['state']=='PENDING_NO_SCORE':assert x['reference_run_index']==2,'DirectGOallowsONLYpendingrun2while1/3/5replace'
  else:assert prior_done['SDK_terminal']
 prior=next(x for x in refs if x['reference_run_index']==a.run); assert prior['state'] in {'PARTIAL123_NOT_ADMISSIBLE','TERMINAL_NO_RESULTS_REQUIRES_PRIMARY_INSPECTION'}; assert prior['score'] is None
 if prior['state']=='PARTIAL123_NOT_ADMISSIBLE':
  pp=Path(prior['partial_inspection']); assert sha(pp.read_bytes())==prior['partial_inspection_sha256']; part=json.loads(pp.read_bytes()); assert part['missing_rewards']>0 and part['scored_binary_rewards']<123 and not part['admissible_complete_run']
 assert not (D/f'run-{a.run}'/'native-doc-reward-arrays-r1.json').exists(),'NEVERretryvalid123score'
 sd=R/f'run-{a.run}'; out=sd/'launch.json'; assert not out.exists(),'ONEreplacementalreadyconsumed;readonlyreconcile'
 load_dotenv('/home/karan/oa/marin/.env'); configure_coreweave_s3()
 for x in proof_data['metadata'].values():
  fs,k=filesystem_for(x['uri']); assert sha(fs.cat_file(k))==x['sha256']
 artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-v01-reference5-replacements-20261030-r1/run-{a.run}'; records=artifact+'/records'; fs,k=filesystem_for(records); assert not fs.exists(k) and not fs.find(k)
 orders=N.read_bytes(); assert sha(orders)==a.orders_sha256 and b'Reference infra retries' in orders and b'ONE full fresh rerun' in orders and b'old 34/123 is NOT pooled' in orders
 argv=list(binding['argv']); argv[argv.index('--model-config')+1]=str(model); argv[argv.index('--version')+1]=f'pathb-v01-reference5-r1-run{a.run}-whole-replacement1'; argv[argv.index('--records-prefix')+1]=records; argv[argv.index('--description')+1]=f'Jerboa2b WHOLEFRESH replacement1 reference slot{a.run}; original incomplete retained; NOpartialrepair/validscoreselection'
 os.environ['MARIN_PREFIX']=artifact+'/native'; os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None); sd.mkdir(parents=True,exist_ok=True)
 record={'state':'PRE_SUBMIT','opened_utc':utc(),'authority':'Jerboa2b+DIRECTdoNOTwaitrun2:WHOLEFRESHreplacements1/3/5NOW;run2ownifconfirmedterminalincomplete;NOpartialrepair/zeros/validscorererun','arm':'v0.1','reference_run_index':a.run,'max_application_attempts_this_order':1,'model_yaml':str(model),'model_yaml_sha256':sha(model.read_bytes()),'HF':proof_data['only_A0_config_deltas']['location']['new'],'source_proof':str(proof),'source_proof_sha256':sha(proof.read_bytes()),'method_pin':'caba0fa38b780af42da54af2f188ba5bfcc28501','records_prefix':records,'argv':argv,'priority':'interactive','sampling_temperature_seed_reward_parser_changes':False,'requested_native_failure_preemption_retries':[0,0],'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'native_checkpoint_ops':0,'training_submits':0,'no_automatic_retry_on_uncertain_submit':True,'historical34_pooled':False}
 record.update(replacement_number=1,replaces_incomplete_job=prior['run_id'],original_incomplete_launch=prior['launch'],original_incomplete_launch_sha256=prior['launch_sha256'],coverage_index=str(a.coverage_index),coverage_index_sha256=sha(a.coverage_index.read_bytes()),all_five_original_coverage_states_recorded=True,pending_original_run2_does_not_block_1_3_5=True,max_new_replacements_per_slot=1,original_partial_outcomes_retained=True)
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
