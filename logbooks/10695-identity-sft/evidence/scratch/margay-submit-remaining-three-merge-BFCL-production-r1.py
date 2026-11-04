"""ONE remaining merge original BFCL slot: sealed M25/M49/M75, isolated production prepare+submit."""
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
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');N=P.parent/'t310/current-orders.md';C=Path('/home/karan/oa/marin-path-b-swe100');PROD=job_pb2.PRIORITY_BAND_PRODUCTION
PIN='43bba4a16d175072b20fea27b8fb5524c377c79f';ZIP='5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b';PACKETS={
'M25':('deaad7a43bc9b5f9033449460974cdcddc164c95ae076c3f0626f5fca86dd5b4','0ddbe1218060c83fd27b8e5e9dfc92a4edfd6ec852340fd8bfef8c5ce1f87282','67ec74f2123e62118f774be72504ea154f70afa069819c31e41894f6070520f9','16714fc1111068a1c0686637c1040d8655627703157e507e9eea7f5c7115f8ee'),
'M49':('f32a480eef1b7397ca77c3b0d2130bcf7747e2f22dcc3816593130c07cf11b10','ecfbde5b1a36d309e44951275f8552a812e92909067e207dd602744f5b143337','56ae60fceb6b5dc81c67edd2a565bf4640916a058f04d60edb2dfe586725eaeb','d37abe32db423d4aba1e63b6f6c8bfeb84ec70ad76ebdd8f9b12b56a04305d41'),
'M75':('6899fb6e9cc0575e9395dcacc4ad39604cc7c715dece1c109572301323a72141','fda6916f356dff3253b7255db6ecd1f37e2aa65dbdd97a167b72e097ddc57543','fc55e2f0f633baa7036a8e7d114c7c2b5373808a533e349d932dd563ad0631f7','00fc7f3bcdd1a68eb344dd8e9fabececba4365e85e943355a43586aa1b3ecb37')}
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=list(PACKETS),required=True);ap.add_argument('--run',type=int,choices=range(1,6),required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);D=P/f'{a.arm}-BFCL-production-five-r1';CLEAR,RECEIPT,PLAN,READY=PACKETS[a.arm]
 sealed={'binding-review-marten-r1.json':CLEAR,'review-receipt.json':RECEIPT,'launch-plan.json':PLAN}
 for name,h in sealed.items():assert sha((D/name).read_bytes())==h
 review=json.loads((D/'binding-review-marten-r1.json').read_bytes());assert review['verdict']==f'CLEAR_{a.arm}_PRODUCTION_BFCL5_TARGET_ONLY'and review['source_pin']==PIN and review['native_bundle_sha256']==ZIP and review['actual_readiness_sha256']==READY
 for path,r in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==r['bytes']and sha(b)==r['sha256']
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert sha(create_workspace_zip(C))==ZIP
 plan=json.loads((D/'launch-plan.json').read_bytes());target=plan['targets'][a.arm];item=next(x for x in target['runs']if x['run']==a.run);assert sha(Path(target['model_config']).read_bytes())==target['model_config_sha256'];assert sha(Path(plan['suite']['path']).read_bytes())==plan['suite']['sha256']=='73bf0dc40d438b04fedaf1ac31bc4c35d3fc07ce851d815730c5079f168666e1';rb=Path(plan['actual_readiness']['path']).read_bytes();assert sha(rb)==plan['actual_readiness']['sha256']==READY;ready=json.loads(rb);assert ready['HF']==target['location']and ready['ready_for_authorized_PRODUCTION_BFCL5_native132_first_original_counts_candidate_only_maths']and ready['provenance_kind']=='linear-cpu-merge'and len(ready['all509_RNO_object_HEADs_match_worker_streamed_full_SHA_manifest'])==509;pr=Path(plan['preregistration']['path']);assert sha(pr.read_bytes())=='108c2f65840ea7ab22acdb677494726a35d316e7400af5ba048337e84d44c4b9';orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'0.51'in orders and b'run evals at PRODUCTION'in orders
 sd=D/'BFCL-five-r1'/f'run-{a.run}';sd.mkdir(parents=True,exist_ok=True);out=sd/'launch.json';assert not out.exists(),'Consumed original slot: reconcile read-only, no repeat';load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 for x in ready['metadata'].values():
  f,k=filesystem_for(x['uri']);b=f.cat_file(k);assert len(b)==x['bytes']and sha(b)==x['sha256']
 for uri in[item['records_prefix'],item['MARIN_PREFIX']]:
  f,k=filesystem_for(uri);assert not f.exists(k)and not f.find(k)
 spec=importlib.util.spec_from_file_location('sealed_remaining_merge_prod_BFCL5',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);batch=m.prepare(a.arm,a.run);assert batch.model.location==target['location']and batch.records_prefix==item['records_prefix']and batch.priority_band==PROD and os.environ['MARIN_PREFIX']==item['MARIN_PREFIX']and len(batch.evaluations)==1;ev=batch.evaluations[0];assert ev.identity.output_dir==item['records_prefix']+'/'+ev.identity.run_id+'/results';proof=json.loads((D/'native-five-bindings-proof.json').read_bytes());pol=next(x for x in proof['all5']if x['run']==a.run)['task_config']['config']['stable_policy_json'];assert ev.executor.config.stable_policy_json==pol
 def owned(b):
  cfg=m.held_bfcl.inference_config_for_model(b.model,b.accelerator,env_vars={},priority=b.priority_band,capability_origin=b.capability_origin,api_model=b.api_model);assert cfg.iris.priority==PROD
 owned(batch);expected='/karan/eval-'+batch.group_id
 record={'state':'PRE_PROBE','utc':utc(),'arm':a.arm,'original_slot':a.run,'replacement_number':0,'authority':f'Standing production {a.arm} BFCL5 on own actual honest merge HF; exact sealed target review, no sibling/science gate','source_pin':PIN,'packet_sha256':RECEIPT,'plan_sha256':PLAN,'review_sha256':CLEAR,'orders_sha256_submission_only':sha(orders),'local_native_workspace_bundle_sha256':ZIP,'expected_job':expected,'HF':target['location'],'HF_readiness_proof':plan['actual_readiness']['path'],'HF_readiness_proof_sha256':READY,'merge_admission_sha256':ready['metadata']['merge-admission.json']['sha256'],'merge_manifest_sha256':ready['metadata']['merge-manifest.json']['sha256'],'no_fake_training_event_OR_master_cast_audit':True,'records_prefix':item['records_prefix'],'MARIN_PREFIX':item['MARIN_PREFIX'],'registered_native_evaluations':[{'run_id':ev.identity.run_id,'output_dir':ev.identity.output_dir,'actual_prepared_output_dir':ev.identity.output_dir}],'statistics_spec_sha256':sha(pr.read_bytes()),'original_all123_before_transcript_read':True,'sampling_parser_operator_source_changes':False,'application_submit_calls':0,'native_checkpoint_operations':0,'credentials_in_receipt':False,'requested_root_AND_owned_serve_priority':'production','child_GLOBAL_mirror_NOT_original_policy_or_independent_priority_proof':True}
 def save():
  with out.open('w')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with out.open('x')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 if a.run==1:
  old=P/'bfcl-failure-diagnostic-r1/diagnostic369-frozen-runs-docIDs-margay-r1.json';assert sha(old.read_bytes())=='08020bffdd9b9c41222c446caededf5f705232178ac131045c7b7ad808e2b4e8';ids=json.loads(old.read_bytes())['runs'][0]['doc_ids'];assert len(ids)==len(set(ids))==123;cp=D/'original-BFCL-slot1-diagnostic-cohort-margay-r1.json';cohort={'arm':a.arm,'original_evaluation_slot':1,'replacement_number':0,'actual_original_job_id_expected_and_guarded':expected,'actual_native_eval_id':ev.identity.run_id,'all123_doc_ids':ids,'launch':str(out),'readiness_sha256':READY,'frozen_before_submit_AND_any_transcript_read':True,'all123_including_passes':True,'missing_original_UNKNOWN_no_firstsuccess_or_replacement_substitution':True}
  with cp.open('x')as h:json.dump(cohort,h,indent=2);h.write('\n')
  record['original_slot1_cohort_sha256']=sha(cp.read_bytes());save()
 body=secrets.token_bytes(32);key=f'tmp/ttl=14d/snowball-identity/path-b/{a.arm}-BFCL-production-submit-probes/{a.run}-'+secrets.token_hex(8)+'.bin';record['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=resolve_secret_spec('env:CW_KEY_ID').value,aws_secret_access_key=resolve_secret_spec('env:CW_KEY_SECRET').value,config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
 try:
  s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);stream=s3.get_object(Bucket='marin-rno2a',Key=key)['Body']
  try:assert stream.read()==body
  finally:stream.close()
 finally:s3.close()
 record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected)
   class Guard:
    def submit(self,**kwargs):
     assert record['application_submit_calls']==0 and N.read_bytes()==orders
     for name,h in sealed.items():assert sha((D/name).read_bytes())==h
     assert kwargs['name']=='eval-'+batch.group_id and kwargs['timeout'].to_seconds()==176400 and kwargs['priority_band']==PROD and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['entrypoint'].workdir_files;fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert len(args)==1 and not kw;actual=args[0];assert actual.model.location==target['location']and actual.records_prefix==item['records_prefix']and actual.priority_band==PROD and actual.evaluations[0].identity.output_dir==ev.identity.output_dir and actual.evaluations[0].executor.config.stable_policy_json==pol and actual.accelerator.target_cluster=='cw-rno2a';owned(actual);assert kwargs['environment'].env_vars['MARIN_PREFIX']==item['MARIN_PREFIX'];assert'cw-rno2a'in json.dumps([MessageToDict(c.to_proto(),preserving_proto_field_name=True)for c in kwargs['constraints']]);record.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc(),root_SDK_EXACT_namespace_and_owned_production_configuration_checked=True,required_callable_workdir_serialization_preserved=True);save()
     try:job=client.submit(**kwargs)
     except Exception as e:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Reconcile read-only, never blind retry')from None
     record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expected;log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();return job
   m.submit(batch,Guard());assert record['application_submit_calls']==1
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request;cs=[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints];record['actual_root_stored_request']={'bundle_id':q.bundle_id,'priority':job_pb2.PriorityBand.Name(q.priority_band),'constraints':cs,'timeout_ms':q.timeout.milliseconds,'failure_retries':q.max_retries_failure,'preemption_retries':q.max_retries_preemption,'MARIN_PREFIX_RPC_redacted':q.environment.env_vars['MARIN_PREFIX']};record['actual_native_binding_PASS']=q.bundle_id==ZIP and q.priority_band==PROD and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000 and'cw-rno2a'in json.dumps(cs)and q.environment.env_vars['MARIN_PREFIX']==redact_value({'MARIN_PREFIX':item['MARIN_PREFIX']})['MARIN_PREFIX'];save()
   except Exception as e:record['native_readback_error_type']=type(e).__name__;save()
 print('MERGE_PRODUCTION_BFCL_ORIGINAL_SUBMITTED',a.arm,a.run,str(out),sha(out.read_bytes()),record['job'],'NATIVE_PASS',record.get('actual_native_binding_PASS'),flush=True);assert record.get('actual_native_binding_PASS')is True,'Already submitted; do not repeat consumed slot'
if __name__=='__main__':main()
