"""One NEW SWE100 logicalslot4/5, isolated prepare+submit; no old-slot revival."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,secrets,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2,job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3,s3_endpoint
from rigging.filesystem.cluster_config import StoreType
from rigging.secrets import resolve_secret_spec
from rigging.redaction import redact_value
from botocore.session import get_session
from botocore.config import Config
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'second-agentic-SWE100-production-topup-r1';E=P/'second-agentic-swe100-production-topup-execution-r1';N=P.parent/'t310/current-orders.md';PROD=job_pb2.PRIORITY_BAND_PRODUCTION
BUNDLE='5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b';RECEIPT='e4d92caf45a1992e6262ec6ed4167f20620a40a3fe62b178cc5b7d5bb78b292c';PREREG='c221c5bb9f1c49f9cb2dd4152481aefd97b61c040d55740d8cc040ce756b6b9d'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A3','A8a'],required=True);ap.add_argument('--run',choices=[4,5],type=int,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();review_bytes=a.review.read_bytes();assert a.review.parent==D and sha(review_bytes)==a.review_sha256;review=json.loads(review_bytes);assert review['verdict']=='CLEAR_NEW_SWE100_A3_A8a_SLOTS4_5_PRODUCTION_ONLY'and review['plan_sha256']==sha((D/'launch-plan.json').read_bytes())and review['receipt_sha256']==RECEIPT and review['native_bundle_sha256']==BUNDLE,'Sealed specific production topup binding CLEAR required';pb=(D/'launch-plan.json').read_bytes();plan=json.loads(pb);receipt_bytes=(D/'review-receipt.json').read_bytes();assert sha(receipt_bytes)==RECEIPT;receipt=json.loads(receipt_bytes)
 for path,row in receipt['files'].items():
  b=Path(path).read_bytes();assert len(b)==row['bytes']and sha(b)==row['sha256'],path
 add=D/'production-priority-clarification-addendum.json';assert sha(add.read_bytes())=='e60ac5ae40099d39d925c5507b934b67c6dc40a1e8b64a79545c521efa349e21';assert plan['priority']=='production'and plan['new_logical_runs_not_replacements'];assert set(plan['targets'])=={'A3','A8a'};runs=[(label,r)for label,t in plan['targets'].items()for r in t['runs']];assert{(label,r['run'])for label,r in runs}=={('A3',4),('A3',5),('A8a',4),('A8a',5)};assert len({r['records_prefix']for _,r in runs})==len({r['MARIN_PREFIX']for _,r in runs})==4
 C=Path(plan['workspace']);os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['code_pin']==review['source']=='43bba4a16d175072b20fea27b8fb5524c377c79f';assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert sha(Path(plan['suite']['path']).read_bytes())==plan['suite']['sha256']=='117cc38b33491a6a843793fb0f43314b7851c01f0da665ff70251fec33eed156';registered=json.loads((P/'second-agentic-swe100-execution-r1/registered12-logical-slots-and-fixed100tasks-margay-r1.json').read_bytes());assert registered['fixed100_task_names']==plan['suite']['task_ids']and len(set(plan['suite']['task_ids']))==100
 for name in['preregistration','cost_choice','topup_preregistration']:assert sha(Path(plan[name]['path']).read_bytes())==plan[name]['sha256']
 assert plan['topup_preregistration']['sha256']==PREREG;orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'SWE top-up'in orders and b'run evals at PRODUCTION'in orders;target=plan['targets'][a.arm];item=next(r for r in target['runs']if r['run']==a.run);assert sha(Path(target['model_config']).read_bytes())==target['model_config_sha256'];sd=E/a.arm/f'topup-run-{a.run}';sd.mkdir(parents=True,exist_ok=True);out=sd/'launch.json';assert not out.exists(),'Consumed journal: reconcile read-only, never blindly rerun'
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 for name in['event','audit']:
  ref=target['guards'][name];f,k=filesystem_for(ref['uri']);assert sha(f.cat_file(k))==ref['sha256']
 for uri in[item['records_prefix'],item['MARIN_PREFIX']]:
  f,k=filesystem_for(uri);assert not f.exists(k)and not f.find(k),'Fresh namespace required'
 assert sha(create_workspace_zip(C))==BUNDLE;spec=importlib.util.spec_from_file_location('sealed_production_SWE_topup',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);batch=mod.prepare(a.arm,a.run);assert batch.model.location==target['location']and len(batch.evaluations)==1 and batch.priority_band==PROD;assert os.environ['MARIN_PREFIX']==item['MARIN_PREFIX']and batch.records_prefix==item['records_prefix'];ev=batch.evaluations[0];assert ev.identity.output_dir==batch.records_prefix+'/'+ev.identity.run_id+'/results';expected='/karan/eval-'+batch.group_id;proof=json.loads((D/'native-four-bindings-proof.json').read_bytes());chosen=next(x for x in proof['all4']if x['label']==a.arm and x['run']==a.run);sealed_policy=chosen['task_config']['config']['stable_policy_json'];assert ev.executor.config.stable_policy_json==sealed_policy and batch.accelerator.target_cluster=='cw-rno2a'
 def owned_priority(b):
  cfg=mod.held_bfcl.inference_config_for_model(b.model,b.accelerator,env_vars={},priority=b.priority_band,capability_origin=b.capability_origin,api_model=b.api_model);assert cfg.iris.priority==PROD;return'production'
 owned_priority(batch);record={'state':'PRE_SUBMIT','utc':utc(),'authority':'Jerboa NEW SWEtopup4/5 production + sealed narrow actual-bindingCLEAR; NOT old-slot retry','arm':a.arm,'original_slot':a.run,'replacement_number':0,'new_topup_logical_slot':True,'source_pin':plan['code_pin'],'plan_sha256':sha(pb),'packet_sha256':RECEIPT,'review_sha256':a.review_sha256,'topup_preregistration_sha256':PREREG,'orders_sha256_submission_only':sha(orders),'local_native_bundle_sha256':BUNDLE,'model_yaml_sha256':target['model_config_sha256'],'model_HF':target['location'],'records_prefix':item['records_prefix'],'MARIN_PREFIX':item['MARIN_PREFIX'],'expected_job':expected,'group_id':batch.group_id,'registered_native_evaluations':[{'run_id':ev.identity.run_id,'eval_name':ev.identity.eval_ref.name,'actual_prepared_output_dir':ev.identity.output_dir}],'all100_original_task_names_frozen_BEFORE_any_transcript_read':plan['suite']['task_ids'],'scientific_config_sha256':plan['suite']['sha256'],'priority_root_and_owned_serve':'production','inherited_policy_requested_priority_is_historical_SUPERSEDED':True,'priority_clarification_sha256':sha(add.read_bytes()),'standard_Harbor_internal_infra_retry_policy_unchanged':plan['policy']['internal_Harbor_retry'],'sampling_agent_parser_renderer_changes':False,'application_submit_calls':0,'no_automatic_retry_uncertain_submit':True,'native_checkpoint_ops':0,'credentials_or_environment_values_in_receipt':False}
 def save():
  with out.open('w')as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 with out.open('x')as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/SWE100-production-topup-probes/'+a.arm+'-'+str(a.run)+'-'+secrets.token_hex(8)+'.bin';record['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=resolve_secret_spec('env:CW_KEY_ID').value,aws_secret_access_key=resolve_secret_spec('env:CW_KEY_SECRET').value,config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
 try:
  s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);stream=s3.get_object(Bucket='marin-rno2a',Key=key)['Body']
  try:assert stream.read()==body
  finally:stream.close()
 finally:s3.close()
 record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected)
   class Guarded:
    def submit(self,**kwargs):
     assert record['application_submit_calls']==0 and N.read_bytes()==orders and sha(a.review.read_bytes())==a.review_sha256;assert kwargs['name']=='eval-'+batch.group_id and kwargs['timeout'].to_seconds()==176400 and kwargs['priority_band']==PROD and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;assert kwargs['entrypoint'].workdir_files;fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert len(args)==1 and not kw;actual=args[0];assert actual.records_prefix==item['records_prefix']and actual.model.location==target['location']and actual.priority_band==PROD;assert actual.evaluations[0].identity.output_dir==ev.identity.output_dir and actual.evaluations[0].executor.config.stable_policy_json==sealed_policy and actual.accelerator.target_cluster=='cw-rno2a';owned_priority(actual);assert kwargs['environment'].env_vars['MARIN_PREFIX']==item['MARIN_PREFIX'];constraints=[MessageToDict(x.to_proto(),preserving_proto_field_name=True)for x in kwargs['constraints']];assert'cw-rno2a'in json.dumps(constraints);record.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc(),required_serialization_preserved=True,actual_root_and_owned_serve_PRODUCTION_checked_at_call=True,root_env_MARIN_PREFIX_checked_at_actual_call=item['MARIN_PREFIX']);save()
     try:job=client.submit(**kwargs)
     except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile read-only: no blind retry')from None
     record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expected;log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();return job
   mod.submit(batch,Guarded());assert record['application_submit_calls']==1
   try:
    native=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job']));q=native.request;record['actual_native_request']={'bundle_id':q.bundle_id,'constraints':[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints],'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'priority':job_pb2.PriorityBand.Name(q.priority_band),'timeout_ms':q.timeout.milliseconds,'actual_root_env_MARIN_PREFIX_RPC_REDACTED':q.environment.env_vars['MARIN_PREFIX'],'namespace_exact_at_SDK_call_not_claimed_injective_RPC':True};record['actual_native_binding_PASS']=q.bundle_id==BUNDLE and q.priority_band==PROD and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000 and q.environment.env_vars['MARIN_PREFIX']==redact_value({'MARIN_PREFIX':item['MARIN_PREFIX']})['MARIN_PREFIX']and'cw-rno2a'in json.dumps(record['actual_native_request']['constraints']);save()
   except Exception as error:record['native_readback_error_type']=type(error).__name__;save()
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'arm':a.arm,'original_slot':a.run,'actual_native_binding_PASS':record.get('actual_native_binding_PASS')}),flush=True);assert record.get('actual_native_binding_PASS')is True,'Already submitted: reconcile, never retry consumed slot'
if __name__=='__main__':main()
