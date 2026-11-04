"""Explicit director exception: final WHOLE R2 for A3slot3/A8aslot1 ONLY. NEVER R3."""
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
from rigging.secrets import resolve_secret_spec
from rigging.redaction import redact_value
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'second-agentic-SWE100-infra-replacement2-r1';E=P/'second-agentic-swe100-execution-r1';N=P.parent/'t310/current-orders.md';R='619cdc7f91ae3dee3daf959e06723953d3d34952cb406fbce3857775aedd7b9f';PL='13cd485344052ef11782abb0a4454a06cc50d2f7aaf2ee730891c801749c1532';ZIP='5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b';PIN='43bba4a16d175072b20fea27b8fb5524c377c79f'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A3','A8a'],required=True);ap.add_argument('--run',type=int,required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();assert(a.arm,a.run)in[('A3',3),('A8a',1)];assert sha((D/'review-receipt.json').read_bytes())==R and sha((D/'launch-plan.json').read_bytes())==PL;plan=json.loads((D/'launch-plan.json').read_bytes());assert plan['replacement_number']==2 and {(arm,x['run'])for arm,t in plan['targets'].items()for x in t['runs']}=={('A3',3),('A8a',1)};C=Path(plan['workspace']);os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 for path,x in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==x['bytes']and sha(b)==x['sha256']
 assert sha(Path(plan['parent_clear']['path']).read_bytes())==plan['parent_clear']['sha256']=='f514fa59aeda5f8f51f9fb18a94a7ed8ea96bea96638b84e30918a3867ca9248';proof_path=Path(plan['prior_replacement1_failure']['path']);pb=proof_path.read_bytes();assert sha(pb)==plan['prior_replacement1_failure']['sha256']=='679b9842618acec130280aae904d49fbeaed899308ca26323f851395a6057f40';case=next(x for x in json.loads(pb)['cases']if(x['arm'],x['original_slot'])==(a.arm,a.run));assert case['result_object_count']==0 and case['native_record_projection']['status']=='infra_failed'and case['native_record_projection']['coverage']=={};assert all(x['state']=='preempted'and'WorkloadEvictedDueToPreempted'in x['error_message']for t in case['serve']['tasks']for x in t['attempts']);target=plan['targets'][a.arm];item=target['runs'][0];assert item['run']==a.run and case['job']==item['prior_replacement1_job']
 for key in ['original','prior_replacement1']:assert sha(Path(item[key+'_launch_receipt']).read_bytes())==item[key+'_launch_sha256']
 assert sha(Path(target['model_config']).read_bytes())==target['model_config_sha256'];assert sha(Path(plan['suite']['path']).read_bytes())==plan['suite']['sha256']=='117cc38b33491a6a843793fb0f43314b7851c01f0da665ff70251fec33eed156';assert len(set(plan['suite']['task_ids']))==100;registered=json.loads((E/'registered12-logical-slots-and-fixed100tasks-margay-r1.json').read_bytes());assert registered['fixed100_task_names']==plan['suite']['task_ids'];sd=E/a.arm/f'replacement-2-for-original-run-{a.run}';sd.mkdir(exist_ok=True);out=sd/'launch.json';assert not out.exists(),'FINALR2 journal consumed; NO R3 or blindretry';orders=N.read_bytes();assert sha(orders)==a.orders_sha256;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 for key in ['event','audit']:
  g=target['guards'][key];f,k=filesystem_for(g['uri']);assert sha(f.cat_file(k))==g['sha256']
 for key in ['records_prefix','MARIN_PREFIX']:
  assert item[key]!=item['original_'+key]and item[key]!=item['prior_replacement1_'+key];f,k=filesystem_for(item[key]);assert not f.exists(k)and not f.find(k),'R2namespace not fresh'
 assert sha(create_workspace_zip(C))==ZIP;spec=importlib.util.spec_from_file_location('sealed_final_SWE_R2',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);batch=m.prepare(a.arm,a.run);assert len(batch.evaluations)==1 and batch.model.location==target['location']and batch.records_prefix==item['records_prefix']and os.environ['MARIN_PREFIX']==item['MARIN_PREFIX'];assert batch.evaluations[0].identity.output_dir==item['records_prefix'].rstrip('/')+'/'+batch.evaluations[0].identity.run_id+'/results';expected='/karan/eval-'+batch.group_id;r={'utc':utc(),'state':'PRE_PROBE','authority':'DIRECT Jerboa explicit ONE more whole R2 ONLY A3slot3/A8aslot1; if third preemption stopslot/usefewerruns. NO automatic R3.','arm':a.arm,'original_slot':a.run,'replacement_number':2,'original_job':item['original_job'],'prior_replacement1_job':item['prior_replacement1_job'],'prior_replacement1_launch_sha256':item['prior_replacement1_launch_sha256'],'prior_replacement1_startup_preemption_proof_sha256':sha(pb),'packet_sha256':R,'plan_sha256':PL,'source_pin':PIN,'native_bundle_sha256':ZIP,'orders_sha256_submission_only':sha(orders),'expected_job':expected,'group_id':batch.group_id,'model_HF':target['location'],'records_prefix':item['records_prefix'],'MARIN_PREFIX':item['MARIN_PREFIX'],'registered_native_evaluations':[{'run_id':e.identity.run_id,'output_dir':e.identity.output_dir}for e in batch.evaluations],'all100_fixed_task_names':plan['suite']['task_ids'],'application_submit_calls':0,'NO_R3_NO_valid100_rerun_NO_partial_repair':True,'all_original_and_R1_outcomes_costs_retained':True,'credentials_in_receipt':False}
 def save():
  with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/SWE100-finalR2-probes/'+a.arm+'-'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=resolve_secret_spec('env:CW_KEY_ID').value,aws_secret_access_key=resolve_secret_spec('env:CW_KEY_SECRET').value,config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
 try:
  s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);resp=s3.get_object(Bucket='marin-rno2a',Key=key);stream=resp['Body']
  try:assert stream.read()==body
  finally:stream.close()
 finally:s3.close()
 r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert client.job_state(JobName.from_string(item['original_job'])).value==client.job_state(JobName.from_string(item['prior_replacement1_job'])).value=='failed';assert not client.list_jobs(prefix=expected)
   class Guard:
    def submit(self,**kw):
     assert r['application_submit_calls']==0 and N.read_bytes()==orders;assert kw['name']=='eval-'+batch.group_id and kw['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kw['max_retries_failure']==kw['max_retries_preemption']==0 and kw['timeout'].to_seconds()==176400;assert kw['entrypoint'].workdir_files;fn,args,kws=kw['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kws);assert len(args)==1 and not kws and args[0].records_prefix==item['records_prefix']and args[0].model.location==target['location']and args[0].evaluations[0].identity.output_dir==batch.evaluations[0].identity.output_dir;assert kw['environment'].env_vars['MARIN_PREFIX']==item['MARIN_PREFIX'];r['SDK_EXACT_unredacted_MARIN_PREFIX']=kw['environment'].env_vars['MARIN_PREFIX'];assert'cw-rno2a'in json.dumps([MessageToDict(x.to_proto(),preserving_proto_field_name=True)for x in kw['constraints']]);r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
     try:job=client.submit(**kw)
     except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Read-only reconcile; NO R3 or blindretry')from None
     r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected;log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save();return job
   m.submit(batch,Guard());assert r['application_submit_calls']==1
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];r['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_ms':q.timeout.milliseconds,'MARIN_PREFIX_RPC_redacted':q.environment.env_vars['MARIN_PREFIX'],'redaction_noninjective_not_exact_runtime_namespace_proof':True};r['actual_native_binding_PASS']=q.bundle_id==ZIP and'cw-rno2a'in json.dumps(cs)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000 and q.environment.env_vars['MARIN_PREFIX']==redact_value({'MARIN_PREFIX':item['MARIN_PREFIX']})['MARIN_PREFIX'];save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':r['job'],'arm':a.arm,'original_slot':a.run,'replacement':2,'actual_native_binding_PASS':r.get('actual_native_binding_PASS'),'NO_R3':True}),flush=True);assert r.get('actual_native_binding_PASS')is True,'Already submitted: reconcileonly, no R3'
if __name__=='__main__':main()
