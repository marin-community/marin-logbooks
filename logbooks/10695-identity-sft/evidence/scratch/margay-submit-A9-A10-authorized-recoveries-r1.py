"""Exactly the two director-preauthorized recoveries, after actual RNO write gate. No new science."""
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
from experiments.post_training.identity_path_b.write_recovery import checked_original_training
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'write-recovery-a9-a10-r1';C=Path('/home/karan/oa/marin-path-b-write-recovery');N=P.parent/'t310/current-orders.md';PIN='36627ae6560aab1e1dc75adc26970436ba23f6eb';ZIP='4b1afda4a23953bf58975656a276e6901f9030dc51d2d9e51f4581ca25b4862c';RECEIPT='f319902be954e42c02599ae811d2b2b73d6a82da058443969ec618089c31f091';PLAN='500eef46584564c4d300a664d876a21c09cd56fe1480685542ec7958a736788c'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
 f,k=filesystem_for(uri);return f.cat_file(k)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['A9','A10'],required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);out=D/a.target/'actual-recovery-launch-margay-r1.json';out.parent.mkdir(exist_ok=True);assert not out.exists(),'Recovery ONCE journal exists: reconcile read-only, never blind retry';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A9/A10 recovery after the write suspension'in orders;gate=P/'cleanup-write-suspension3-r1/ONE-probe-monitor-FIRST-SUCCESS.json';gb=gate.read_bytes();assert sha(gb)=='bb9ee999bd64852f6e0fb1dd66a7926fca76385aaf7f9763417e3496aee1f802'and json.loads(gb)['state']=='PUT_FULLGET_32B_BYTE_EQUAL_PASS';assert sha((D/'review-receipt.json').read_bytes())==RECEIPT and sha((D/'launch-plan.json').read_bytes())==PLAN;plan=json.loads((D/'launch-plan.json').read_bytes());item=plan['arms'][a.target];assert item['event']['arm']==a.target and item['event']['completed_updates']==(72 if a.target=='A9' else 36);assert item['mode']==('EXPORT_ONLY_FROM_ORIGINAL_NATIVE71'if a.target=='A9'else'FRESH_FULL_RETRAIN_FROM_V01');assert sha(Path(plan['runtime']['local']).read_bytes())==plan['runtime']['sha256'];load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k)for k in ['CW_KEY_ID','CW_KEY_SECRET']}
 source=[]
 for key in ['source_receipt','source_objects']:
  v=item['training'][key];b=get(v['path']);assert sha(b)==v['sha256'];source.append({'uri':v['path'],'bytes':len(b),'sha256':sha(b)})
 for pathkey,hashkey in [('packed_path','packed_sha256'),('exposure_path','exposure_sha256')]:
  b=get(item['training'][pathkey]);assert sha(b)==item['training'][hashkey];source.append({'uri':item['training'][pathkey],'bytes':len(b),'sha256':sha(b)})
 if a.target=='A9':checked_original_training()
 spec=importlib.util.spec_from_file_location('sealed_director_storage_recovery',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);kwargs=m.request(a.target,credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert kwargs['entrypoint'].workdir_files,'Native callable serializedfiles MUSTremain';assert kwargs['name']==item['job_name']and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400 and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE
 if a.target=='A9':assert fn.__name__=='export_only_worker'and not args and set(kw)=={'runtime_bytes'}
 else:assert kw['training']['arm']=='A10'and kw['training']['run_root']==item['root']and kw['training']['training_pin']=='cf4da6f40326df150eac712e2eb323667f71a267'
 assert sha(create_workspace_zip(C))==ZIP;f,k=filesystem_for(item['root']);assert not f.exists(k)and not f.find(k),'Recovery root notfresh: no overwrite/retry';expected_job='/karan/'+item['job_name']
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected_job);r={'utc':utc(),'state':'PRE_PROBE','target':a.target,'mode':item['mode'],'authority':'Jerboa preauthorizedA9exportONLYfromnative71+A10freshunchangedretrain, currentorders; exactexporter/science, NOnewreviewgate','orders_sha256_submission_only':sha(orders),'code_pin':PIN,'review_receipt_sha256':RECEIPT,'plan_sha256':PLAN,'expected_native_bundle_sha256':ZIP,'write_gate_sha256':sha(gb),'expected_job':expected_job,'new_run_root':item['root'],'new_HF':item['event']['hf_export'],'original_native':item['event']['native_checkpoint']if a.target=='A9'else None,'actual_original_A9_metadata3e609_rechecked':a.target=='A9','source_and_data_FULLGET':source,'native_callable_serialization_preserved':True,'single_export_GPU8_2h_CPUprobe_before_allocation_then502audit_exactnativeretire':True,'no_original_failed_HF_overwrite_or_partial_admission':True,'standing_followups':'ownactualreadinessthenBFCL5+originalslot1diagnostic+ONLYnativeboundary132once','application_submit_calls':0,'automatic_retry':False,'credentials_in_receipt':False}
   def save():
    with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/write-recovery-submit-probes-36627/'+a.target+'-'+secrets.token_hex(8)+'.bin';r['fresh_write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
   try:
    s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);resp=s3.get_object(Bucket='marin-rno2a',Key=key);stream=resp['Body']
    try:assert stream.read()==body
    finally:stream.close()
   finally:s3.close()
   r['fresh_write_probe']['state']='PUT_FULLGET_32B_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders and sha((D/'review-receipt.json').read_bytes())==RECEIPT;r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Read-only reconcile ONLY, no blind retry')from None
   r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected_job;log=(out.parent/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(out.parent/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save()
   try:
    raw=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job']));q=raw.request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];r['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority_band':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_milliseconds':q.timeout.milliseconds,'resources':MessageToDict(q.resources,preserving_proto_field_name=True)};r['actual_native_binding_PASS']=q.bundle_id==ZIP and 'cw-rno2a'in json.dumps(cs)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000;save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
   print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':r['job'],'one_notifier_pid':proc.pid,'actual_native_binding_PASS':r.get('actual_native_binding_PASS')}),flush=True)
if __name__=='__main__':main()
