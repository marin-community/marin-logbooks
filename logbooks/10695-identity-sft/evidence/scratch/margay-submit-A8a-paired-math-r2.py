"""ONE independent math-only owner per target on its narrow sealed CLEAR."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,secrets,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2,job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
from experiments.post_training.identity_fullscreen_owner.math_only import ARCHIVE_PATH,check_model
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'A8a-paired-math-r2';C=Path('/home/karan/oa/marin-path-b-a8a-paired-math-r2');N=P.parent/'t310/current-orders.md';PIN='4f49e3b016fb8731a19a3e9b3e2d06d65b7a9370';BUNDLE='ea871b57ee2aa7e9a079a6086d0a4688e266f484730461071fb2f20411574951'
REQUIRED_VERDICT='CLEAR_TWO_MATH_ONLY_OWNERS_A8A_AND_FRESH_V01'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A8a','v01'],required=True);ap.add_argument('--orders-sha256',required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);a=ap.parse_args();os.chdir(C);out=D/f'{a.arm}-actual-launch-margay-r1.json';assert not out.exists(),'Consumed launch journal: reconcile read-only, never retry blindly'
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 expected={'receipt.json':'4f4e4f291597244e77ffdea6601c64bd8349125a9f2d0ad2dfb7a99a028f2d57','launch-plan.json':'97ab3bca9afc1c567dcf0fac1e153fc1b017629f4b35ff4e0f8dbccc88e35236','native-binding-proof.json':'1ce6b32dcbcedc5a68c6698cfcb13c430ddd54788dcbee04faff0ede9fdd962e'}
 for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
 assert sha(a.review.read_bytes())==a.review_sha256=='009ae9ac36a9e152a09f3f292914ce1f55beea011697affae659a4ad5977ba86';review=json.loads(a.review.read_bytes());assert review['verdict']==REQUIRED_VERDICT and review['source_pin']==PIN and review['plan_sha256']==expected['launch-plan.json'] and review['receipt_sha256']==expected['receipt.json'] and review['native_root_bundle_sha256']==BUNDLE
 assert sha((D/'independent-binding-proof-marten-r1.json').read_bytes())==review['independent_proof_sha256']=='9ce161df9ad73620a2ca8ab0b0fca8edca00b5191d1948b831e6dcd2c3d91a99'
 plan=json.loads((D/'launch-plan.json').read_bytes());target=plan['targets'][a.arm];candidate=target['candidate'];assert candidate['scope']=='math-only' and [t['trial'] for t in candidate['math']]==['greedy','17','29'];assert plan['method']['GSM8K']['thinking'] is False and plan['method']['MATH500']['thinking'] is True
 for row in json.loads((D/'receipt.json').read_bytes())['files']:
  raw=Path(row['path']).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
 assert sha(Path(plan['preregistration']['path']).read_bytes())==plan['preregistration']['sha256']=='ff96537805d31ee55f5f5704a7e53139229245c6dd177bac57939ef249eb6c78';archive=Path(plan['source_archive']['path']).read_bytes();assert sha(archive)==plan['source_archive']['sha256'];orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A8a maths' in orders
 spec=importlib.util.spec_from_file_location('sealed_math_r2',D/'prepare_request.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ('CW_KEY_ID','CW_KEY_SECRET')};check_model(candidate);kwargs=module.request(a.arm,credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs)
 assert kwargs['name']==target['root_name'] and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400
 assert set(kwargs['entrypoint'].workdir_files)=={ARCHIVE_PATH} and kwargs['entrypoint'].workdir_files[ARCHIVE_PATH]==archive;constraints=[MessageToDict(x.to_proto(),preserving_proto_field_name=True) for x in kwargs['constraints']];assert 'cw-rno2a' in json.dumps(constraints)
 bundle=create_workspace_zip(C);assert sha(bundle)==BUNDLE;fs,key=filesystem_for(candidate['artifact_root']);assert not fs.exists(key) and not fs.find(key),'Output not fresh: never overwrite';expected_job='/karan/'+target['root_name']
 with connect_controller(cluster_name='marin') as connection:
  with IrisClient.remote(connection.url,workspace=C,credentials=connection.credentials) as client:
   assert not client.list_jobs(prefix=expected_job)
   record={'state':'PRE_PROBE','utc':utc(),'arm':a.arm,'authority':'Jerboa A8a pairedmath + narrow Marten bindingCLEAR; independent of A9/baseline','code_pin':PIN,'sealed_files':expected,'review':str(a.review),'review_sha256':a.review_sha256,'preregistration_sha256':plan['preregistration']['sha256'],'orders_sha256_submission_only':sha(orders),'expected_job':expected_job,'artifact_root':candidate['artifact_root'],'candidate_sha256':target['candidate_sha256'],'model':candidate['model_path'],'method':plan['method'],'local_native_bundle_sha256':BUNDLE,'local_native_bundle_bytes':len(bundle),'large_f5_workdir_archive':{'path':ARCHIVE_PATH,'sha256':sha(archive),'bytes':len(archive),'preserved_not_removed':True},'f5_runtime_bundle_identity':'Fresh extraction ZIP timestamps vary: no f001 expected runtime-ID assertion; pinned archive and all4759 member content bindings per9ce161; actual runtime IDs recorded separately','resources':plan['resources'],'constraints':constraints,'model_guards_full_GET_before_submit':candidate['model_guard'],'application_submit_calls':0,'automatic_retry':False,'checkpoint_export_native_weight_lease_or_delete_ops':False,'credentials_in_receipt':False}
   def save():
    with out.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with out.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   probe=secrets.token_bytes(32);uri='s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/paired-math-r2-submit-probes-4f49e3b016fb/'+a.arm+'-'+secrets.token_hex(8)+'.bin';fs,key=filesystem_for(uri);assert not fs.exists(key);record['write_probe']={'uri':uri,'bytes':32,'sha256':sha(probe),'state':'PUT_CALLED'};save();fs.pipe_file(key,probe);assert fs.cat_file(key)==probe;record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
   assert N.read_bytes()==orders and sha(a.review.read_bytes())==a.review_sha256
   for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
   record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile read-only; no automatic retry') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expected_job
   log=(D/f'{a.arm}-completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/f'{a.arm}-completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save()
   try:
    native=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job']));q=native.request;actual_constraints=[MessageToDict(x,preserving_proto_field_name=True) for x in q.constraints];actual={'bundle_id':q.bundle_id,'constraints':actual_constraints,'priority_band':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_milliseconds':q.timeout.milliseconds,'state':job_pb2.JobState.Name(native.job.state),'run_argv':list(q.entrypoint.run_command.argv),'resource_spec':MessageToDict(q.resources,preserving_proto_field_name=True)};record['actual_native_request']=actual;record['actual_native_binding_PASS']=q.bundle_id==BUNDLE and 'cw-rno2a' in json.dumps(actual_constraints) and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000;save()
   except Exception as error:record['native_readback_error_type']=type(error).__name__;save()
   print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'actual_native_binding_PASS':record.get('actual_native_binding_PASS'),'one_notifier_pid':proc.pid}),flush=True)
if __name__=='__main__':main()
