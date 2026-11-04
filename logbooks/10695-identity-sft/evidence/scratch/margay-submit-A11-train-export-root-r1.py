"""Exactly one fresh A11 OFF-only72 root on combined source/data/native CLEAR."""
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
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-a11-r1';C=Path('/home/karan/oa/marin-path-b-a11-off72');N=P.parent/'t310/current-orders.md';PIN='741e5928030cdc0e2a4b0f0199ad3caf881e60c3';BUNDLE='b39217ca18f595795c2083ff5fdd382fd4d988c82fcc27a576dcf7eae6a5917f';OUT=D/'actual-training-export-launch-margay-r1.json';CLEAR='ccebd8701187d6a51eed7a15e344b048321758e03b4c0391b26b89edc3e35b39'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
 f,k=filesystem_for(uri);return f.cat_file(k)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);assert not OUT.exists(),'Consumed journal: reconcile read-only; never blind retry';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();expected={'review-receipt.json':'153edb66065e3d35fb7187f64560d4b3565466e2b2cea43082999559b41027c8','launch-plan.json':'dc12a12a8a94dec3182a958a35642c8befb1c3f06e44dfa54f6dd1d0c6fee947','combined-verdict-marten-r1.json':CLEAR}
 for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
 review=json.loads((D/'combined-verdict-marten-r1.json').read_bytes());assert review['verdict']=='CLEAR_A11_COMBINED_SOURCE_DATA_NATIVE_BINDING'and review['source_pin']==PIN and review['receipt_sha256']==expected['review-receipt.json']and review['plan_sha256']==expected['launch-plan.json']and review['native_bundle_sha256']==BUNDLE
 for name,digest in [('independent_data_native_proof','b8ac65f7531c96991888abbe9cc4e5e36bd04b72c42d7b3e1420f68bd9e8f8d2'),('independent_final_binding_proof','f561a50ac51b725a8bb66ba84bb4d614358ed0b7b6e9301ebdab761981a8e293')]:assert sha((D/review[name]).read_bytes())==review[name+'_sha256']==digest
 for path,x in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  raw=Path(path).read_bytes();assert len(raw)==x['bytes']and sha(raw)==x['sha256']
 plan=json.loads((D/'launch-plan.json').read_bytes());assert len(plan['arms'])==1 and plan['training']['updates']==72 and plan['training']['learning_rate']==2e-6;target=plan['arms'][0];assert target['arm']=='A11'and target['checkpoint_updates']==[72]and not target['restore_from_checkpoint']and target['frozen_native_tensors']==104 and target['trained_master_cast_tensors']==398 and target['special_rows_frozen']and target['padding_only_sequences']==0 and target['all72_batches_nonempty_supervision']and target['selected_ON_occurrences']==0;assert target['final_checkpoint']['native_checkpoint'].endswith('/step-71')and target['final_checkpoint']['completed_updates']==72
 for name in ['current_input_manifest','current_staging_receipt','runtime']:assert sha(Path(plan[name].get('path',plan[name].get('local'))).read_bytes())==plan[name]['sha256']
 stage=json.loads(Path(plan['current_staging_receipt']['path']).read_bytes());assert stage['state']=='ALL3_FULLGET_BYTE_SHA_LENGTH_PASS_DATA_ONLY'and len(stage['files'])==3 and stage['data_bytes']==8702732;orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A11 (GO) = A3 recipe at 72 updates'in orders;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k)for k in('CW_KEY_ID','CW_KEY_SECRET')}
 for item in stage['files']:
  raw=get(item['uri']);assert len(raw)==item['bytes']and sha(raw)==item['sha256']
 for key in ['source_receipt','source_objects']:assert sha(get(plan[key]['path']))==plan[key]['sha256']
 spec=importlib.util.spec_from_file_location('sealed_A11_root',D/'prepare_request.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);kwargs=module.request(credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);fn,args,call_kwargs=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**call_kwargs);assert not args;training=call_kwargs['training'];assert training['arm']=='A11'and training['training_pin']==PIN and'resume'not in training and training['runtime_bytes']==Path(plan['runtime']['local']).read_bytes();assert kwargs['entrypoint'].workdir_files,'Preserve mandatory native callable serialization';assert kwargs['name']==target['root_name']and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400;bundle=create_workspace_zip(C);assert sha(bundle)==BUNDLE;run_root=training['run_root'];f,k=filesystem_for(run_root);assert not f.exists(k)and not f.find(k);expected_job='/karan/'+target['root_name']
 with connect_controller(cluster_name='marin')as connection:
  with IrisClient.remote(connection.url,workspace=C,credentials=connection.credentials)as client:
   assert not client.list_jobs(prefix=expected_job);record={'state':'PRE_PROBE','utc':utc(),'authority':'Jerboa A11 GO + Marten ccebd870 combinedCLEAR; independent of A9/A10/maths/archive/SWE','code_pin':PIN,'sealed_files':expected,'orders_sha256_submission_only':sha(orders),'expected_job':expected_job,'run_root':run_root,'staging_sha256':plan['current_staging_receipt']['sha256'],'local_native_bundle_sha256':BUNDLE,'native_bundle_bytes':len(bundle),'required_callable_workdir_serialization_preserved':True,'fresh_v01_no_resume':True,'completed_updates_requested':72,'only_final_native_step':71,'actual_dose':target['actual_dose'],'dose_comparison':plan['dose_comparison'],'limits_reported_before_submit':plan['comparison_limits'],'consumed_occurrences':14995,'selected_identity_ON_occurrences':0,'first36_data_and_LR_equal_A3_NOT_empirical_bit_identical_training_claim':True,'NOT_A9_seven_padding_fill':True,'CPU_probe_before_single8H100_2h_export_then502audit_exact_TEMP_native71_retirement':True,'standing_followups':'OWN actually auditedHF then5BFCL+ONE nativeONLY132+fixed ORIGINALslot1 bbebcounts+candidateONLY pairedmaths/reuseadmittedv01sixunits; NOoriginalparserextra/newrefserve','application_submit_calls':0,'automatic_retry':False,'credentials_in_receipt':False}
   def save():
    with OUT.open('w')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with OUT.open('x')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   probe=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/A11-submit-probes/'+secrets.token_hex(8)+'.bin';record['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(probe),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
   try:
    s3.put_object(Bucket='marin-rno2a',Key=key,Body=probe);resp=s3.get_object(Bucket='marin-rno2a',Key=key);stream=resp['Body']
    try:assert stream.read()==probe
    finally:stream.close()
   finally:s3.close()
   record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders
   for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
   record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Read-only reconciliation only; no blind retry')from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expected_job;log=(D/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save()
   try:
    native=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job']));q=native.request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];record['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority_band':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_milliseconds':q.timeout.milliseconds,'resources':MessageToDict(q.resources,preserving_proto_field_name=True),'state':job_pb2.JobState.Name(native.job.state)};record['actual_native_binding_PASS']=q.bundle_id==BUNDLE and'cw-rno2a'in json.dumps(cs)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000;save()
   except Exception as error:record['native_readback_error_type']=type(error).__name__;save()
   print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),'job':record['job'],'one_notifier_pid':proc.pid,'actual_native_binding_PASS':record.get('actual_native_binding_PASS')}),flush=True)
if __name__=='__main__':main()
