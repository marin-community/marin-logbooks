"""Five ordered original HF-only BFCL runs on actual fresh A9 recovery; isolated prepare+submit per run."""
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
from rigging.secrets import resolve_secret_spec
from rigging.redaction import redact_value
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'A9-recovered-BFCL-five-r1';C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing');N=P.parent/'t310/current-orders.md';R='09469f52b4312ae9ddd89e501e24b3a0a60e0554fa81acc7a659da74f7bdffae';PL='a7bcbbdda41c0c5653cf2970b87bcf51c60c0b52fd7bfb44224f74427d27af76';PIN='caba0fa38b780af42da54af2f188ba5bfcc28501'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=int,choices=range(1,6),required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);assert sha((D/'review-receipt.json').read_bytes())==R and sha((D/'launch-plan.json').read_bytes())==PL;assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN
 for path,x in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==x['bytes']and sha(b)==x['sha256']
 plan=json.loads((D/'launch-plan.json').read_bytes());target=plan['targets']['A9'];item=next(x for x in target['runs']if x['run']==a.run);assert sha(Path(target['model_config']).read_bytes())==target['model_config_sha256'];assert sha(Path(plan['suite']['path']).read_bytes())==plan['suite']['sha256'];assert sha(Path(plan['preregistration']['path']).read_bytes())==plan['preregistration']['sha256']=='0aa09019d739a0d106d11acb09334dc9b2827200f192573f9f5e89237cc9aa77';ready_path=Path(plan['actual_readiness']['path']);rb=ready_path.read_bytes();assert sha(rb)==plan['actual_readiness']['sha256']=='ecd3e7968d80e4bfc0c8bbf841919672f4dd999524c3aa197d1c0222ecb5518e';ready=json.loads(rb);assert ready['arm']=='A9'and ready['HF']==target['location']and ready['actual_bfcl_ready_marker'];orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A9/A10 recovery'in orders;sd=D/'BFCL-five-r1'/f'run-{a.run}';sd.mkdir(parents=True,exist_ok=True);out=sd/'launch.json';assert not out.exists(),'Original logicalslot consumed; no valid123 rerun/blindretry';load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 for x in ready['metadata'].values():
  f,k=filesystem_for(x['uri']);b=f.cat_file(k);assert len(b)==x['bytes']and sha(b)==x['sha256']
 for key in ['records_prefix','MARIN_PREFIX']:
  f,k=filesystem_for(item[key]);assert not f.exists(k)and not f.find(k),'Original BFCL namespace not fresh'
 spec=importlib.util.spec_from_file_location('sealed_A9_recovered_BFCL',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);batch=m.prepare('A9',a.run);assert len(batch.evaluations)==1 and batch.model.location==target['location']and batch.records_prefix==item['records_prefix']and os.environ['MARIN_PREFIX']==item['MARIN_PREFIX'];assert batch.evaluations[0].identity.output_dir==item['records_prefix'].rstrip('/')+'/'+batch.evaluations[0].identity.run_id+'/results';bundle_sha=sha(create_workspace_zip(C));expected='/karan/eval-'+batch.group_id;r={'utc':utc(),'state':'PRE_PROBE','arm':'A9','original_slot':a.run,'replacement_number':0,'authority':'Standing A9 BFCL5 after actual fresh export-only recovery502audit; unchangedCABA/HarborPi123, no extra science/sibling wait','packet_sha256':R,'plan_sha256':PL,'source_pin':PIN,'local_native_workspace_bundle_sha256':bundle_sha,'orders_sha256_submission_only':sha(orders),'expected_job':expected,'HF':target['location'],'HF_readiness_proof':str(ready_path),'HF_readiness_proof_sha256':sha(rb),'event_sha256':ready['metadata']['event.json']['sha256'],'audit_sha256':ready['metadata']['audit.json']['sha256'],'records_prefix':item['records_prefix'],'MARIN_PREFIX':item['MARIN_PREFIX'],'registered_native_evaluations':[{'run_id':e.identity.run_id,'output_dir':e.identity.output_dir}for e in batch.evaluations],'statistics_spec_sha256':plan['preregistration']['sha256'],'fixed_original_slot1_diagnostic_spec':'bbebfea8faa31d21cd5c570e453cf58c839a78f9d1d306ce87ab6109a5772d85','original_all123_before_transcript_read':True,'temperature_seed_parser_operator_source_changes':False,'native_checkpoint_operations':0,'application_submit_calls':0,'one_whole_fresh_only_after_terminal_incomplete_original':True,'NO_valid123_rerun_partial_repair_zero_imputation_or_best_of':True,'credentials_in_receipt':False}
 def save():
  with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/A9-recovered-BFCL-submit-probes/'+str(a.run)+'-'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=resolve_secret_spec('env:CW_KEY_ID').value,aws_secret_access_key=resolve_secret_spec('env:CW_KEY_SECRET').value,config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
 try:
  s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);resp=s3.get_object(Bucket='marin-rno2a',Key=key);stream=resp['Body']
  try:assert stream.read()==body
  finally:stream.close()
 finally:s3.close()
 r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected)
   class Guard:
    def submit(self,**kw):
     assert r['application_submit_calls']==0 and N.read_bytes()==orders;assert kw['name']=='eval-'+batch.group_id and kw['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kw['max_retries_failure']==kw['max_retries_preemption']==0 and kw['timeout'].to_seconds()==176400;assert kw['entrypoint'].workdir_files;fn,args,kws=kw['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kws);assert len(args)==1 and not kws and args[0].records_prefix==item['records_prefix']and args[0].model.location==target['location']and args[0].evaluations[0].identity.output_dir==batch.evaluations[0].identity.output_dir;assert kw['environment'].env_vars['MARIN_PREFIX']==item['MARIN_PREFIX'];r['SDK_EXACT_unredacted_MARIN_PREFIX']=kw['environment'].env_vars['MARIN_PREFIX'];assert'cw-rno2a'in json.dumps([MessageToDict(x.to_proto(),preserving_proto_field_name=True)for x in kw['constraints']]);r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
     try:job=client.submit(**kw)
     except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Read-only reconciliation; no blind retry')from None
     r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected;log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save();return job
   m.submit(batch,Guard());assert r['application_submit_calls']==1
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];r['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_ms':q.timeout.milliseconds,'MARIN_PREFIX_RPC_redacted':q.environment.env_vars['MARIN_PREFIX'],'redaction_noninjective_not_exact_runtime_namespace_proof':True};r['actual_native_binding_PASS']=q.bundle_id==bundle_sha and'cw-rno2a'in json.dumps(cs)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000 and q.environment.env_vars['MARIN_PREFIX']==redact_value({'MARIN_PREFIX':item['MARIN_PREFIX']})['MARIN_PREFIX'];save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
 if a.run==1:
  old=P/'bfcl-failure-diagnostic-r1/diagnostic369-frozen-runs-docIDs-margay-r1.json';assert sha(old.read_bytes())=='08020bffdd9b9c41222c446caededf5f705232178ac131045c7b7ad808e2b4e8';ids=json.loads(old.read_bytes())['runs'][0]['doc_ids'];assert len(ids)==len(set(ids))==123;cohort={'arm':'A9','original_evaluation_slot':1,'replacement_number':0,'actual_original_job_id':r['job'],'actual_native_eval_id':r['registered_native_evaluations'][0]['run_id'],'launch':str(out),'launch_sha256':sha(out.read_bytes()),'readiness_sha256':sha(rb),'doc_ids':ids,'preregistration_sha256':'bbebfea8faa31d21cd5c570e453cf58c839a78f9d1d306ce87ab6109a5772d85','frozen_before_any_session_read':True,'all123_including_passes':True,'missing_original_explicit_no_first_success_or_replacement_substitution':True};cp=D/'original-BFCL-slot1-diagnostic-cohort-margay-r1.json'
  with cp.open('x')as h:json.dump(cohort,h,indent=2);h.write('\n')
  print('FROZEN_ORIGINAL_SLOT1',str(cp),sha(cp.read_bytes()),flush=True)
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':r['job'],'original_slot':a.run,'actual_native_binding_PASS':r.get('actual_native_binding_PASS')}),flush=True);assert r.get('actual_native_binding_PASS')is True,'Already submitted: reconcile only before others'
if __name__=='__main__':main()
