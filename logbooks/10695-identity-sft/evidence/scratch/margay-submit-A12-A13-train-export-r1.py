"""One fresh A12 or A13 training/export root on combined actual-binding CLEAR; no retries."""
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
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-a12-a13-r1';C=Path('/home/karan/oa/marin-path-b-a12-a13');N=P.parent/'t310/current-orders.md';PIN='69ddc88fb2ff4a6e45ee5cbbec866d9f86c2d068';BUNDLE='7b790ba0ef4f3d61623a3e6b2388f90b2b29cfd85c906b7a4de23d26f6644c85';CLEAR='e079f407e707a95d93aa459bbdf1b5136e11a86e7a4b4e9d06f70c124dbabec5';RECEIPT='84ce9e36f5647e59b7a3a763f47673263249db23b12f77e3c248743855d1fde9';PLAN='366915edeb29e9a4b1243f74bc7fd9ae55dc8957ab14f256bd56ceb351b5086c'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
 f,k=filesystem_for(uri);return f.cat_file(k)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A12','A13'],required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);sd=D/a.arm;sd.mkdir(exist_ok=True);out=sd/'actual-training-export-launch-margay-r1.json';assert not out.exists(),'Consumed journal: read-only reconcile, no blind retry';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();expected={'review-receipt.json':RECEIPT,'launch-plan.json':PLAN,'combined-review-marten-r1.json':CLEAR}
 for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
 review=json.loads((D/'combined-review-marten-r1.json').read_bytes());assert review['verdict']=='CLEAR_A12_A13_COMBINED_DATA_SCIENCE_NATIVE_TRAIN_EXPORT_BINDING'and review['source_pin']==PIN and review['receipt_sha256']==RECEIPT and review['plan_sha256']==PLAN and review['native_bundle_sha256']==BUNDLE;assert sha(Path(review['independent_proof']['path']).read_bytes())==review['independent_proof']['sha256']=='3d242e390940a2c2dd5ae48f41fa418c1e6b603c676cf0000d8066da889ed2cf'
 for path,x in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  raw=Path(path).read_bytes();assert len(raw)==x['bytes']and sha(raw)==x['sha256']
 plan=json.loads((D/'launch-plan.json').read_bytes());assert{r['arm']for r in plan['arms']}=={'A12','A13'};target=next(r for r in plan['arms']if r['arm']==a.arm);updates={'A12':18,'A13':36}[a.arm];assert target['updates']==plan['training']['updates_by_arm'][a.arm]==updates and target['checkpoint_updates']==[updates]and not target['restore_from_checkpoint']and target['frozen_native_tensors']==104 and target['trained_master_cast_tensors']==398 and target['special_rows_frozen']and target['padding_only_sequences']==0 and target['all_batches_nonempty_supervision'];assert target['final_checkpoint']['native_checkpoint'].endswith('/step-'+str(updates-1))and target['final_checkpoint']['completed_updates']==updates;assert plan['training']['learning_rate']==2e-6 and plan['training']['warmup']==2 and plan['training']['schedule']=='constant'and plan['training']['priority']=='interactive'and target['followup_eval_priority']=='production';assert plan['training']['hosts']==4 and plan['training']['gpus_per_host']==8
 for name in['current_input_manifest','current_staging_receipt','runtime','analysis_preregistration']:assert sha(Path(plan[name].get('path',plan[name].get('local'))).read_bytes())==plan[name]['sha256']
 stage=json.loads(Path(plan['current_staging_receipt']['path']).read_bytes());assert stage['state']=='ALL6_FULLGET_BYTE_SHA_LENGTH_PASS_DATA_ONLY'and len(stage['files'])==6 and stage['data_bytes']==7392938;orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A12 = A8a at 18 updates'in orders and b'A13 = A8a + thinking-ON tool replay x2'in orders;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k)for k in('CW_KEY_ID','CW_KEY_SECRET')}
 for item in stage['files']:
  raw=get(item['uri']);assert len(raw)==item['bytes']and sha(raw)==item['sha256']
 for key in['source_receipt','source_objects']:assert sha(get(plan[key]['path']))==plan[key]['sha256']
 spec=importlib.util.spec_from_file_location('sealed_A12_A13_root',D/'prepare_request.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);kwargs=module.request(a.arm,credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);fn,args,callkwargs=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**callkwargs);assert not args;training=callkwargs['training'];assert training['arm']==a.arm and training['training_pin']==PIN and'resume'not in training and training['runtime_bytes']==Path(plan['runtime']['local']).read_bytes();assert kwargs['entrypoint'].workdir_files and kwargs['name']==target['root_name']and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400;bundle=create_workspace_zip(C);assert sha(bundle)==BUNDLE;root=training['run_root'];f,k=filesystem_for(root);assert not f.exists(k)and not f.find(k);expectedjob='/karan/'+target['root_name'];constraints=[MessageToDict(x.to_proto(),preserving_proto_field_name=True)for x in kwargs['constraints']];assert'cw-rno2a'in json.dumps(constraints)
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expectedjob);record={'state':'PRE_PROBE','utc':utc(),'arm':a.arm,'authority':'Jerboa freshA12/A13 GO + e079 combinedCLEAR; independent of Mini/Merge/SWE/sibling','code_pin':PIN,'sealed_files':expected,'orders_sha256_submission_only':sha(orders),'expected_job':expectedjob,'run_root':root,'staging_sha256':plan['current_staging_receipt']['sha256'],'local_native_bundle_sha256':BUNDLE,'native_bundle_bytes':len(bundle),'required_callable_workdir_serialization_preserved':True,'fresh_v01_no_resume':True,'completed_updates_requested':updates,'only_final_native_step':updates-1,'actual_dose':target['actual_dose'],'dose_comparison':plan['dose_comparison'],'limits_reported_before_submit':plan['comparison_limits'],'selected_identity_ON_occurrences':target['selected_ON_occurrences'],'training_priority':'interactive','all_future_eval_priorities':'production_root_AND_owned_serve','NO_illustrative_BFCL_argv_execution':True,'CPU_probe_before_single8H100_2h_export_then502audit_exact_TEMP_native_retirement':True,'standing_followups':'OWN actual auditedHF then5BFCL+ONE nativeONLY132+fixed ORIGINALslot1counts+candidateONLY pairedmaths/reuseadmittedv01six;NOextraoriginalparser/newrefserve','application_submit_calls':0,'automatic_retry':False,'credentials_in_receipt':False}
   def save():
    with out.open('w')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with out.open('x')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/A12-A13-submit-probes/'+a.arm+'-'+secrets.token_hex(8)+'.bin';record['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
   try:
    s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);stream=s3.get_object(Bucket='marin-rno2a',Key=key)['Body']
    try:assert stream.read()==body
    finally:stream.close()
   finally:s3.close()
   record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders
   for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
   record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Read-only reconcile: no blind retry')from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expectedjob;log=(sd/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(sd/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save()
   try:
    native=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job']));q=native.request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];record['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority_band':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_milliseconds':q.timeout.milliseconds,'resources':MessageToDict(q.resources,preserving_proto_field_name=True),'state':job_pb2.JobState.Name(native.job.state)};record['actual_native_binding_PASS']=q.bundle_id==BUNDLE and'cw-rno2a'in json.dumps(cs)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000;save()
   except Exception as error:record['native_readback_error_type']=type(error).__name__;save()
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'arm':a.arm,'one_notifier':proc.pid,'actual_native_binding_PASS':record.get('actual_native_binding_PASS')}),flush=True);assert record.get('actual_native_binding_PASS')is True,'Already submitted: reconcile, never repeat consumed call'
if __name__=='__main__':main()
