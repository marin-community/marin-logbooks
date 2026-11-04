"""ONE cleared published-v01 original132 baseline; no trained-event fiction."""
import argparse,datetime,hashlib,importlib.util,json,os,secrets,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
from experiments.post_training.identity_path_b.identity_scope.source_baseline import checked_source
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'identity-v01-baseline-r1';C=Path('/home/karan/oa/marin-path-b-identity-v01');N=P.parent/'t310/current-orders.md';OUT=D/'actual-launch-margay-r1.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C)
 assert not OUT.exists(),'Consumed exclusive launch journal: reconcile read-only, never retry blindly'
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='1ffad15de6e32f75f12cdc9e283263dcc4cb7c42';assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 expected={'receipt.json':'4ea71e2720de24c43198fbd889a0cc8c0fdf595f0913edcee1f98626785fe9b6','launch-plan.json':'8da3c0f65eb7fee53524d98c50cb4f6c7d67414c9bdfb5a6546615a0c4b3032b','science-verdict-marten-r1.json':'36d1a39762813d4e7b6a700d7e20fb44fc77e37be621f82b5dc15ef5a18905ee','independent-binding-proof-marten-r1.json':'59f66d7297b3a46eacbf2740d7e52138c789c58524345ff9f293eee919910564','native-binding-proof.json':'941edd5d300723caca5c8cfb46da7f82b1fd30d2e2744172dbecdff9494f13cb'}
 for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
 plan=json.loads((D/'launch-plan.json').read_bytes());review=json.loads((D/'science-verdict-marten-r1.json').read_bytes());assert review['verdict']=='CLEAR_ONE_PUBLISHED_V01_ORIGINAL_IDENTITY132_BASELINE_ONLY';assert review['source_pin']==plan['code_pin'] and review['receipt_sha256']==expected['receipt.json'] and review['plan_sha256']==expected['launch-plan.json'];assert review['native_bundle_sha256']=='5d4a535f3a45fe281af76247f8e9095bf6bfb48beb1619c9eb1a216af12e969e'
 assert sha((P/'bfcl-noise-r1/v01-identity132-existing-table-check-marten-r1.json').read_bytes())=='e68725ff28f00419b37415d2bc7c2c888ff61947c136ce2b4dfab7c1384ab4cd'
 for row in json.loads((D/'receipt.json').read_bytes())['files']:
  raw=Path(row['path']).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
 orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'v0.1 on identity-132' in orders
 spec=importlib.util.spec_from_file_location('sealed_plain_v01_baseline',D/'prepare_request.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ('CW_KEY_ID','CW_KEY_SECRET')};source=checked_source(plan['root_name']);assert source['model']==plan['model'];kwargs=module.request(credentials);assert kwargs['name']==plan['root_name'] and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and not kwargs['entrypoint'].workdir_files
 f,k=filesystem_for(plan['output']);assert not f.exists(k) and not f.find(k),'Output exists: no rerun/overwrite';expected_job='/karan/'+plan['root_name']
 with connect_controller(cluster_name='marin') as connection:
  with IrisClient.remote(connection.url,workspace=C,credentials=connection.credentials) as client:
   assert not client.list_jobs(prefix=expected_job)
   record={'state':'PRE_PROBE','utc':utc(),'authority':'Jerboa conditionalONCE + Marten36d1CLEAR; noA9/mathdependency','code_pin':plan['code_pin'],'sealed_files':expected,'expected_native_bundle_sha256':review['native_bundle_sha256'],'orders_sha256_submission_only':sha(orders),'expected_job':expected_job,'output':plan['output'],'actual_source_metadata_FULLGET':source,'protocol':plan['protocol'],'application_submit_calls':0,'automatic_retry':False,'trained_event_export_audit_retirement_or_native_checkpoint_ops':False,'credentials_in_receipt':False}
   def save():
    with OUT.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with OUT.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   probe=secrets.token_bytes(32);uri='s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/v01-baseline-submit-probes-1ffad15de6e3/'+secrets.token_hex(8)+'.bin';f,k=filesystem_for(uri);assert not f.exists(k);record['write_probe']={'uri':uri,'bytes':32,'sha256':sha(probe),'state':'PUT_CALLED'};save();f.pipe_file(k,probe);assert f.cat_file(k)==probe;record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
   assert N.read_bytes()==orders
   for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
   record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile read-only; never automatically retry') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expected_job
   log=(D/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),'job':record['job'],'notifier_pid':proc.pid}),flush=True)
if __name__=='__main__':main()
