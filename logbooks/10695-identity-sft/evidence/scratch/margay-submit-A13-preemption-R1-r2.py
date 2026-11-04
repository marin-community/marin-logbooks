"""ONE fresh unchanged A13 training preemptionR1; corrected namespace, no resume."""
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
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'A13-preemption-replacement1-r2';C=Path('/home/karan/oa/marin-path-b-a12-a13');N=P.parent/'t310/current-orders.md';PIN='69ddc88fb2ff4a6e45ee5cbbec866d9f86c2d068';BUNDLE='7b790ba0ef4f3d61623a3e6b2388f90b2b29cfd85c906b7a4de23d26f6644c85';CLEAR='6b5c76bba8beaf638c09986acbb4daf7ffa6d3ccb2012f400cf72ea444f1bf93';RECEIPT='056740384db8021a7e06b4c76977f51098a6f0bc26c818c222a429002216a46b';PLAN='def394f244562687b3889740dfe118e0fdc3128e069bdb3efe47e336f7d14e51'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
 f,k=filesystem_for(uri);return f.cat_file(k)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);out=D/'actual-fresh-preemption-R1-launch-margay-r1.json';assert not out.exists(),'Consumed journal: reconcile read-only, never blind retry';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();expected={'binding-review-marten-r1.json':CLEAR,'review-receipt.json':RECEIPT,'launch-plan.json':PLAN}
 for name,h in expected.items():assert sha((D/name).read_bytes())==h
 review=json.loads((D/'binding-review-marten-r1.json').read_bytes());assert review['verdict']=='CLEAR_ONE_FRESH_A13_PREEMPTION_REPLACEMENT1_NAMESPACE_BINDING'and review['source_pin']==PIN and review['receipt_sha256']==RECEIPT and review['plan_sha256']==PLAN and review['native_bundle_sha256']==BUNDLE;ref=review['independent_review'];assert sha(Path(ref['path']).read_bytes())==ref['sha256']=='dfe4b334a424935fb4ce96180e5eb99f5af102c34717893e0bcaa96aeffe1475'
 for path,r in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==r['bytes']and sha(b)==r['sha256']
 plan=json.loads((D/'launch-plan.json').read_bytes());target=plan['arm'];assert target['arm']=='A13'and target['updates']==36 and target['checkpoint_updates']==[36]and not target['restore_from_checkpoint']and target['frozen_native_tensors']==104 and target['trained_master_cast_tensors']==398 and target['special_rows_frozen']and target['padding_only_sequences']==0 and target['all_batches_nonempty_supervision']and target['followup_eval_priority']=='production';assert target['final_checkpoint']['native_checkpoint'].endswith('/step-35')and target['final_checkpoint']['completed_updates']==36
 for field in['original_plan','original_review','original_failure','original_coverage_clarification','staging']:assert sha(Path(plan[field]['path']).read_bytes())==plan[field]['sha256']
 assert sha(Path(plan['runtime']['local']).read_bytes())==plan['runtime']['sha256'];old=plan['original_training_kwargs'];new=target['training_kwargs'];assert set(old)==set(new)and{k for k in old if old[k]!=new[k]}=={'run_id','run_root'};assert new['run_root']=='s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-a13-preemption-replacement1-20261030-r1/A13/run'and'/bfcl-ablation-'in new['run_root'];assert target['root_name']=='pathb-bfcl-a13-r1-5a3bd99de0a6'and new['run_id']==target['root_name'];stage=json.loads(Path(plan['staging']['path']).read_bytes());assert stage['state']=='ALL6_FULLGET_BYTE_SHA_LENGTH_PASS_DATA_ONLY'and stage['data_bytes']==7392938;original=P/'ablation-a12-a13-r1/A13';done=json.loads((original/'completion-native.json').read_bytes());assert done['SDK_terminal']and done['state']=='failed';orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'Training preemption rule'in orders;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k)for k in['CW_KEY_ID','CW_KEY_SECRET']}
 for r in stage['files']:
  b=get(r['uri']);assert len(b)==r['bytes']and sha(b)==r['sha256']
 for name in['source_receipt','source_objects']:assert sha(get(new[name]['path']))==new[name]['sha256']
 original_launch=json.loads((original/'actual-training-export-launch-margay-r1.json').read_bytes());f,k=filesystem_for(original_launch['run_root']+'/hf');assert not f.exists(k)and not f.find(k),'Original auditedHF would require ruling, not retraining';f,k=filesystem_for(new['run_root']);assert not f.exists(k)and not f.find(k);assert sha(create_workspace_zip(C))==BUNDLE;spec=importlib.util.spec_from_file_location('sealed_A13_preemptionR1',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kwargs=mod.request(credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert not args and kwargs['entrypoint'].workdir_files;assert kw['training']=={**new,'runtime_bytes':Path(plan['runtime']['local']).read_bytes()};assert kwargs['name']==target['root_name']and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400;cs=[MessageToDict(c.to_proto(),preserving_proto_field_name=True)for c in kwargs['constraints']];assert'cw-rno2a'in json.dumps(cs);expectedjob='/karan/'+target['root_name']
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expectedjob);r={'state':'PRE_PROBE','utc':utc(),'arm':'A13','physical_replacement_number':1,'authority':'Jerboa ONE freshunchangedtrainingpreemptionR1 + 6b5c operationalCLEAR; technicalpacket r2 NOTsecondattempt','original_job':original_launch['job'],'original_failure_sha256':plan['original_failure']['sha256'],'original_GPU_missing_windows_NOTzero':True,'source_pin':PIN,'sealed_files':expected,'local_native_bundle_sha256':BUNDLE,'orders_sha256_submission_only':sha(orders),'expected_job':expectedjob,'run_root':new['run_root'],'target_HF':target['final_checkpoint']['hf_export'],'actual_dose':target['actual_dose'],'staging_sha256':plan['staging']['sha256'],'fresh_v01_no_resume_same36':True,'required_callable_workdir_serialization_preserved':True,'training_priority':'interactive','future_eval_root_AND_owned_serve_priority':'production','application_submit_calls':0,'automatic_second_training_retry':False,'credentials_in_receipt':False}
   def save():
    with out.open('w')as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   with out.open('x')as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/A13-R1-submit-probes/'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
   try:
    s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);stream=s3.get_object(Bucket='marin-rno2a',Key=key)['Body']
    try:assert stream.read()==body
    finally:stream.close()
   finally:s3.close()
   r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders
   for name,h in expected.items():assert sha((D/name).read_bytes())==h
   r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as exc:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(exc).__name__);save();raise RuntimeError('Reconcile read-only: no blind retry')from None
   r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expectedjob;log=(D/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(D/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save()
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints];r['actual_native_request']={'bundle_id':q.bundle_id,'priority':job_pb2.PriorityBand.Name(q.priority_band),'constraints':cs,'resources':MessageToDict(q.resources,preserving_proto_field_name=True),'timeout_ms':q.timeout.milliseconds,'failure_retries':q.max_retries_failure,'preemption_retries':q.max_retries_preemption};r['actual_native_binding_PASS']=q.bundle_id==BUNDLE and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and'cw-rno2a'in json.dumps(cs)and q.timeout.milliseconds==176400000 and q.max_retries_failure==q.max_retries_preemption==0;save()
   except Exception as exc:r['native_readback_error_type']=type(exc).__name__;save()
 print('A13_FRESH_PREEMPTION_R1_SUBMITTED',str(out),sha(out.read_bytes()),r['job'],'NATIVE_PASS',r.get('actual_native_binding_PASS'),flush=True);assert r.get('actual_native_binding_PASS')is True,'Already submitted: never repeat consumed call'
if __name__=='__main__':main()
