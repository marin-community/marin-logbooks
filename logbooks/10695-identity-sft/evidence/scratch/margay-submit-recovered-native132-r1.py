"""One native132 per ACTUAL own admitted recovered HF; no original-parser extra generation."""
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
from experiments.post_training.identity_path_b.native_boundary_scope.owner import checked_identity
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'native-boundary-recovered-a9-a10-r1';C=Path('/home/karan/oa/marin-path-b-native-boundary-recovered');N=P.parent/'t310/current-orders.md';PIN='0c15bfbe2918c88e60c76371dee04c7bff52a449';ZIP='bd4004494d932971e1b36919248064922823f17bad5c63c482487dbedb9b96cd';R='b6e561da202cb4f9475e65ba3d958bcc47aa608e927e3f4065270c46c346e570';PL='093b3e3451fc1a289654edce534d2ac8d7d031a40bc3dc65f325f1f45b9ec27e'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A9','A10'],required=True);ap.add_argument('--readiness-sha256',required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);sd=D/a.arm;sd.mkdir(exist_ok=True);out=sd/'actual-launch-margay-r1.json';assert not out.exists(),'One native screen journal consumed; reconcile only';assert sha((D/'review-receipt.json').read_bytes())==R and sha((D/'launch-plan.json').read_bytes())==PL;assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 for path,x in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==x['bytes']and sha(b)==x['sha256']
 plan=json.loads((D/'launch-plan.json').read_bytes());item=plan['targets'][a.arm];pp=P/'write-recovery-a9-a10-r1'/a.arm/'actual-audited-HF-ready-margay-r1.json';rb=pp.read_bytes();assert sha(rb)==a.readiness_sha256;ready=json.loads(rb);assert ready['arm']==a.arm and ready['HF']==item['model'];orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'run ONLY the native-boundary identity-132'in orders;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k)for k in['CW_KEY_ID','CW_KEY_SECRET']}
 for x in ready['metadata'].values():
  f,k=filesystem_for(x['uri']);b=f.cat_file(k);assert len(b)==x['bytes']and sha(b)==x['sha256']
 spec=importlib.util.spec_from_file_location('sealed_recovered_native132',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);audit_sha=m.checked_readiness(a.arm,pp,a.readiness_sha256);actual=checked_identity(a.arm,item['root_name'],audit_sha);kwargs=m.request(a.arm,credentials,pp,a.readiness_sha256);inspect.signature(IrisClient.submit).bind(object(),**kwargs);assert kwargs['name']==item['root_name']and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400;assert not kwargs['entrypoint'].workdir_files,'Native command root no callable payload';assert sha(create_workspace_zip(C))==ZIP;f,k=filesystem_for(item['output']);assert not f.exists(k)and not f.find(k);expected='/karan/'+item['root_name']
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected);r={'utc':utc(),'state':'PRE_PROBE','arm':a.arm,'authority':'Jerboa standing ONLY-native132 afterOWN actual fresh recoveredHF; exact conditionald0b7/62550method, no newscience or sibling wait','expected_job':expected,'output':item['output'],'HF':ready['HF'],'code_pin':PIN,'packet_sha256':R,'plan_sha256':PL,'native_bundle_sha256':ZIP,'orders_sha256_submission_only':sha(orders),'actual_readiness_receipt':str(pp),'actual_readiness_sha256':sha(rb),'actual_audit_sha256':audit_sha,'actual_source_HF_metadata_FULLGET':actual,'request_delta_ONLY':{'skip_special_tokens':False,'return_token_ids':True},'no_P1_no_added_START_no_identity_system':True,'generatedIDs_authoritative_FIRSTgenerated128003_noEndNoONfinalCredit':True,'all_raw_HTTP_provider_fields_usage_finish_IDS_preserved':True,'legacy_proxy_from_SAME_WHOLE_generatedIDs_specials_removed_NOT_postEND_only':True,'NO_separate_original_parser_workflow':True,'application_submit_calls':0,'automatic_retry':False,'credentials_in_receipt':False}
   def save():
    with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/recovered-native132-submit-probes/'+a.arm+'-'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
   try:
    s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);resp=s3.get_object(Bucket='marin-rno2a',Key=key);stream=resp['Body']
    try:assert stream.read()==body
    finally:stream.close()
   finally:s3.close()
   r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders and sha((D/'review-receipt.json').read_bytes())==R;r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Read-only reconciliation only, no blind retry')from None
   r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected;log=(sd/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(sd/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save()
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];r['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority_band':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_milliseconds':q.timeout.milliseconds};r['actual_native_binding_PASS']=q.bundle_id==ZIP and'cw-rno2a'in json.dumps(cs)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000;save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
   print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':r['job'],'one_notifier_pid':proc.pid,'actual_native_binding_PASS':r.get('actual_native_binding_PASS')}),flush=True)
if __name__=='__main__':main()
