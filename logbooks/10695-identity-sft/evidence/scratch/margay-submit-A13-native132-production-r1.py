"""ONE actual fresh36 R1 A13 native132 production root; preserved original payload/prefix method."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,secrets,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from botocore.session import get_session
from botocore.config import Config
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2,job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3,s3_endpoint
from rigging.filesystem.cluster_config import StoreType
from rigging.secrets import require_secret
from experiments.post_training.identity_path_b.native_boundary_scope.owner import checked_identity,service_config
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'native-boundary-A13-r1';C=Path('/home/karan/oa/marin-path-b-native132-a13');N=P.parent/'t310/current-orders.md'
PIN='0c9445ab1eae4f712ee3d264a42277a87bc3ce68';ZIP='74ae32f38fef14198327f68affdc6b4777047d8c506fc2ae48f5d0fe975dccbb';R='bd779c163c5d36a7b8cb3fb32a9702af34680158b8ff69e5f3d73a00d21ff2e1';PL='0bfe5788f4243b04e18a55e1beb1e29f62f2fef10d1cbb13880ea0de8bae843a';CLEAR='52e3f9610395e80e9f7185b0492ddd3a19a19a97449e66b06eeef15d6eb361dd';READY='f82c3b89dbce9a61beefc5571d04809239b2514b3eccfb0df22a42ba2656f21c'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);sd=D/'A13';sd.mkdir(exist_ok=True);out=sd/'actual-launch-margay-r1.json';assert not out.exists();sealed={'review-receipt.json':R,'launch-plan.json':PL,'binding-review-marten-r1.json':CLEAR}
 for name,h in sealed.items():assert sha((D/name).read_bytes())==h
 review=json.loads((D/'binding-review-marten-r1.json').read_bytes());assert review['verdict']=='CLEAR_ONE_A13_ACTUAL_FRESH36_R1_NATIVE132_PRODUCTION'and review['code_pin']==PIN and review['native_bundle_sha256']==ZIP and review['actual_readiness_sha256']==READY
 for path,ref in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==ref['bytes']and sha(b)==ref['sha256']
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert sha(create_workspace_zip(C))==ZIP;plan=json.loads((D/'launch-plan.json').read_bytes());item=plan['targets']['A13'];ref=plan['actual_readiness']['A13'];rb=Path(ref['path']).read_bytes();assert sha(rb)==ref['sha256']==READY;ready=json.loads(rb);assert ready['HF']==item['model']and ready['arm']=='A13'and item['event']['completed_updates']==36;orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A13 = A8a'in orders and b'run evals at PRODUCTION'in orders
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k)for k in ['CW_KEY_ID','CW_KEY_SECRET']}
 for ref in ready['metadata'].values():
  f,k=filesystem_for(ref['uri']);b=f.cat_file(k);assert len(b)==ref['bytes']and sha(b)==ref['sha256']
 actual=checked_identity('A13',item['root_name'],item['actual_audit_sha256']);serve=service_config('A13',item['root_name']);assert serve.iris.priority==job_pb2.PRIORITY_BAND_PRODUCTION and serve.model.weights==item['model']
 spec=importlib.util.spec_from_file_location('sealed_A13_native132',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);kwargs=m.request('A13',credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);assert kwargs['name']==item['root_name']and kwargs['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION and kwargs['timeout'].to_seconds()==176400 and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and not kwargs['entrypoint'].workdir_files
 f,k=filesystem_for(item['output']);assert not f.exists(k)and not f.find(k);expected='/karan/'+item['root_name']
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected);r={'utc':utc(),'state':'PRE_PROBE','arm':'A13','authority':'Standing ONE production native132 after actual fresh36 R1 HF502admission; 52e3 narrowCLEAR/no extra trainingreplicate/sibling/maths gate','expected_job':expected,'HF':item['model'],'output':item['output'],'code_pin':PIN,'sealed_files':sealed,'native_bundle_sha256':ZIP,'orders_sha256_submission_only':sha(orders),'actual_readiness_receipt':ref['path'],'actual_readiness_sha256':READY,'actual_audit_sha256':item['actual_audit_sha256'],'actual_fresh36_R1_event':item['event'],'actual_source_HF_metadata_guard':actual,'request_delta_ONLY':{'skip_special_tokens':False,'return_token_ids':True},'no_P1_no_added_START_no_identity_system':True,'first_generated128003_only_missingEND_noONfinalcredit':True,'raw_HTTP_provider_usage_finish_IDS_preserved':True,'legacy_proxy_same_wholeIDs_specialsremoved_NOT_extra_call':True,'offered_root_AND_owned_serve_PRODUCTION':True,'global_child_mirror_NOT_original_peer_policy_or_priority_proof':True,'application_submit_calls':0,'automatic_retry':False,'credentials_in_receipt':False}
   def save():
    with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/A13-native132-production-submit-probes/'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
   try:
    s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);stream=s3.get_object(Bucket='marin-rno2a',Key=key)['Body']
    try:assert stream.read()==body
    finally:stream.close()
   finally:s3.close()
   r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders
   for name,h in sealed.items():assert sha((D/name).read_bytes())==h
   r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Reconcile read-only; no blind retry')from None
   r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected;log=(sd/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(sd/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save()
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints];r['actual_root_stored_request']={'bundle_id':q.bundle_id,'priority':job_pb2.PriorityBand.Name(q.priority_band),'constraints':cs,'timeout_ms':q.timeout.milliseconds,'failure_retries':q.max_retries_failure,'preemption_retries':q.max_retries_preemption};r['actual_native_binding_PASS']=q.bundle_id==ZIP and q.priority_band==job_pb2.PRIORITY_BAND_PRODUCTION and'cw-rno2a'in json.dumps(cs)and q.timeout.milliseconds==176400000 and q.max_retries_failure==q.max_retries_preemption==0;save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
 print('A13_NATIVE132_PRODUCTION_ACTUALLY_SUBMITTED',str(out),sha(out.read_bytes()),r['job'],'NATIVE_PASS',r.get('actual_native_binding_PASS'),flush=True);assert r.get('actual_native_binding_PASS')is True,'Already submitted; do not repeat consumed call'
if __name__=='__main__':main()
