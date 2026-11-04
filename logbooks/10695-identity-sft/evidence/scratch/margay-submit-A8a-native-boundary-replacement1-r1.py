"""Exactly ONE authorized unchanged A8a native132 replacement after zero-output load infra failure."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,secrets,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from botocore.session import get_session
from botocore.config import Config
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2,job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3,s3_endpoint
from rigging.filesystem.cluster_config import StoreType
from rigging.secrets import require_secret
from experiments.post_training.identity_path_b.native_boundary_scope.owner import checked_identity
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'native-boundary-A8a-replacement1-r1';C=Path('/home/karan/oa/marin-path-b-native-boundary132');N=P.parent/'t310/current-orders.md';PIN='62550b5c905e63e57121576edf8806419e76d0bb';ZIP='726ef4e2b205ca0376f29ec06484d05c6ab7e8b67febc8c634df44e48dac26c7';R='c56ef5f87d44b5f8b0d117c9bfcffdfa19e7a901248762a5cbd6c57a74c67458';PL='bf6d2a8e150bf8cb875aadc6ae8e88b1ad969541a5ea6bae6740fbd64e2bb288'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);out=D/'actual-replacement1-launch-margay-r1.json';assert not out.exists(),'Oneauthorizedreplacementconsumed; nosecond/blindretry';assert sha((D/'review-receipt.json').read_bytes())==R and sha((D/'launch-plan.json').read_bytes())==PL;plan=json.loads((D/'launch-plan.json').read_bytes());item=plan['targets']['A8a'];assert plan['replacement_number']==1 and sha(Path(plan['original_failure_receipt']).read_bytes())==plan['original_failure_sha256']=='bf91d363aa6bd6e218b0d74e30d4573e3c978c7cff711e7c56c0f7fe48b8a0f1';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'Standing infra-rerun rule'in orders
 for p,x in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(p).read_bytes();assert len(b)==x['bytes']and sha(b)==x['sha256']
 assert sha((P/'native-boundary-identity132-r1/science-verdict-marten-r1.json').read_bytes())=='d4c5e457a234b3c48a554ec73de98d349521f8856c50fb8806c768e273937598';load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();creds={k:require_secret(k)for k in ['CW_KEY_ID','CW_KEY_SECRET']};identity=checked_identity('A8a',item['root_name']);spec=importlib.util.spec_from_file_location('sealed_A8a_native_replacement1',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);kwargs=m.request('A8a',creds);inspect.signature(IrisClient.submit).bind(object(),**kwargs);assert kwargs['name']==item['root_name']and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400 and not kwargs['entrypoint'].workdir_files;assert sha(create_workspace_zip(C))==ZIP;f,k=filesystem_for(item['output']);assert not f.exists(k)and not f.find(k);f,k=filesystem_for(item['original_output']);original_keys=f.find(k);assert len(original_keys)==2 and not any('/raw/'in key or key.endswith('generations.jsonl')for key in original_keys);assert json.loads(f.cat_file(k.rstrip('/')+'/status.json'))['status']=='failed';expected='/karan/'+item['root_name'];original='/karan/'+item['original_root_name']
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert client.job_state(JobName.from_string(original)).value=='failed'and not client.list_jobs(prefix=expected);r={'utc':utc(),'state':'PRE_PROBE','target':'A8a','replacement_number':1,'authority':'JerboaDIRECTONCEunchangedA8anativeloadinfrareplace + standingevalzero-scoresrule; no secondreplacement','original_job':original,'original_failure_sha256':plan['original_failure_sha256'],'original_generation_records':0,'original_metadata_objects_preserved':2,'packet_sha256':R,'plan_sha256':PL,'code_pin':PIN,'native_bundle_sha256':ZIP,'orders_sha256_submission_only':sha(orders),'expected_job':expected,'output':item['output'],'actual_source_HF_FULLGET':identity,'method':'Exact62550/d4c5/726ef, same132 originalprompts/nativeprefix/b3/decode; onlyskipfalse+returnIDStrue, noP1/nosystem; preserveallproviderraw/IDs/firstgeneratedENDviews','application_submit_calls':0,'automatic_retry':False,'no_second_replacement':True,'credentials_in_receipt':False}
   def save():
    with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/native132-replacement1-probes/'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=creds['CW_KEY_ID'],aws_secret_access_key=creds['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
   try:
    s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);resp=s3.get_object(Bucket='marin-rno2a',Key=key);stream=resp['Body']
    try:assert stream.read()==body
    finally:stream.close()
   finally:s3.close()
   r['write_probe']['state']='PUT_FULLGET_32B_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders and sha((D/'review-receipt.json').read_bytes())==R;r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('ReadOnlyreconcile, neverblindretry')from None
   r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected;log=(D/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(D/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save()
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];r['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_ms':q.timeout.milliseconds};r['actual_native_binding_PASS']=q.bundle_id==ZIP and'cw-rno2a'in json.dumps(cs)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000;save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
   print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':r['job'],'one_notifier_pid':proc.pid,'actual_native_binding_PASS':r.get('actual_native_binding_PASS')}),flush=True)
if __name__=='__main__':main()
