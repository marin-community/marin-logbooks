"""One conditional WHOLE SWE100 replacement1; terminal-incomplete+director cause gate, isolated prepare/submit."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,secrets,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
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
from botocore.session import get_session
from botocore.config import Config
from rigging.redaction import redact_value
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'second-agentic-SWE100-infra-replacement1-r1';E=P/'second-agentic-swe100-execution-r1';N=P.parent/'t310/current-orders.md';CLEAR='f514fa59aeda5f8f51f9fb18a94a7ed8ea96bea96638b84e30918a3867ca9248';PACKET='dcf83a031d932c7e1844f43357da6fb4d17af85d4a1db16210a18d55da846c78';PLAN='3c4cee88236753bd4ebf0d0298fb7e9d6e6ccbbff7715a065d0e8a73d9839ed7';ZIP='5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['v01','A0','A3','A8a'],required=True);ap.add_argument('--run',choices=[1,2,3],type=int,required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();ad=E/'seven-terminal-incomplete-originals-one-whole-replacement-admission-margay-r1.json';adraw=ad.read_bytes();assert sha(adraw)=='a9b29824843ab5db8db19ff8224f900db605486bb57c34ccb9c3dd6fbb323023';admit=next(x for x in json.loads(adraw)['slots']if(x['arm'],x['original_slot'])==(a.arm,a.run));assert admit['eligible_under_f514_terminal_incomplete_ONE_whole_fresh_rule'];cause=E/'two-serve-cause-direct-Iris-preemption-admission-margay-r1.json';cb=cause.read_bytes();assert sha(cb)=='0aaebcc9af8c09a1b92050d8bed2d0112f578f01f1bc64c1f815b9d3b3e8ab57'and json.loads(cb)['state']=='BOTH_DIRECT_REQUESTED_CASES_CONFIRMED_KUEUE_PREEMPTION_INFRA_CAUSE';assert sha((D/'review-receipt.json').read_bytes())==PACKET and sha((D/'launch-plan.json').read_bytes())==PLAN;plan=json.loads((D/'launch-plan.json').read_bytes());C=Path(plan['workspace']);os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['code_pin']=='43bba4a16d175072b20fea27b8fb5524c377c79f'and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert sha(Path(plan['parent_clear']['path']).read_bytes())==plan['parent_clear']['sha256']==CLEAR
 for path,x in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==x['bytes']and sha(b)==x['sha256']
 assert sha(Path(plan['suite']['path']).read_bytes())==plan['suite']['sha256']=='117cc38b33491a6a843793fb0f43314b7851c01f0da665ff70251fec33eed156';assert len(set(plan['suite']['task_ids']))==100;slots=json.loads((E/'registered12-logical-slots-and-fixed100tasks-margay-r1.json').read_bytes());assert slots['fixed100_task_names']==plan['suite']['task_ids'];target=plan['targets'][a.arm];item=next(x for x in target['runs']if x['run']==a.run);assert item['original_job']==admit['original_job'];assert sha(Path(item['original_launch_receipt']).read_bytes())==item['original_launch_sha256'];assert sha(Path(target['model_config']).read_bytes())==target['model_config_sha256'];sd=E/a.arm/f'replacement-1-for-original-run-{a.run}';sd.mkdir(parents=True,exist_ok=True);out=sd/'launch.json';assert not out.exists(),'ONEreplacement journal consumed; never secondreplacement/blindretry';orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'Standing infra-rerun rule'in orders;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 def get(uri):
  f,k=filesystem_for(uri);return f.cat_file(k)
 guards=target['guards']
 for key in (['source_receipt','source_objects']if guards['kind']=='published_source'else['event','audit']):assert sha(get(guards[key]['uri']))==guards[key]['sha256']
 for uri in [item['records_prefix'],item['MARIN_PREFIX']]:
  f,k=filesystem_for(uri);assert not f.exists(k)and not f.find(k),'Replacement namespace notfresh'
 assert item['records_prefix']!=item['original_records_prefix']and item['MARIN_PREFIX']!=item['original_MARIN_PREFIX'];assert sha(create_workspace_zip(C))==ZIP;spec=importlib.util.spec_from_file_location('sealed_SWE_whole_replacement1',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);batch=m.prepare(a.arm,a.run);assert batch.model.location==target['location']and len(batch.evaluations)==1 and batch.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and os.environ['MARIN_PREFIX']==item['MARIN_PREFIX']and batch.records_prefix==item['records_prefix'];assert batch.evaluations[0].identity.output_dir==item['records_prefix'].rstrip('/')+'/'+batch.evaluations[0].identity.run_id+'/results';expected='/karan/eval-'+batch.group_id;r={'state':'PRE_SUBMIT','utc':utc(),'authority':'StandingONEwholefreshSWEterminalIncomplete + director2casePREEMPTIONcheck satisfied, unchangedf514method','arm':a.arm,'original_slot':a.run,'replacement_number':1,'original_job':item['original_job'],'original_launch_sha256':item['original_launch_sha256'],'terminal_incomplete_admission_sha256':sha(adraw),'two_case_infra_cause_proof_sha256':sha(cb),'source_pin':plan['code_pin'],'plan_sha256':PLAN,'packet_sha256':PACKET,'parent_clear_sha256':CLEAR,'orders_sha256_submission_only':sha(orders),'native_bundle_sha256':ZIP,'model_HF':target['location'],'records_prefix':item['records_prefix'],'MARIN_PREFIX':item['MARIN_PREFIX'],'expected_job':expected,'group_id':batch.group_id,'registered_native_evaluations':[{'run_id':e.identity.run_id,'output_dir':e.identity.output_dir}for e in batch.evaluations],'all100_fixed_task_names':plan['suite']['task_ids'],'application_submit_calls':0,'no_second_replacement_no_valid100_rerun_no_partial_reward_repair':True,'original_partial_outcomes_retained':True,'credentials_in_receipt':False}
 def save():
  with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 body=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/SWE100-replacement1-probes/'+a.arm+'-'+str(a.run)+'-'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=resolve_secret_spec('env:CW_KEY_ID').value,aws_secret_access_key=resolve_secret_spec('env:CW_KEY_SECRET').value,config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
 try:
  s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);resp=s3.get_object(Bucket='marin-rno2a',Key=key);stream=resp['Body']
  try:assert stream.read()==body
  finally:stream.close()
 finally:s3.close()
 r['write_probe']['state']='PUT_FULLGET_32B_BYTE_EQUAL_PASS';save()
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert client.job_state(JobName.from_string(item['original_job'])).value=='failed';assert not client.list_jobs(prefix=expected)
   class Guard:
    def submit(self,**kw):
     assert r['application_submit_calls']==0 and N.read_bytes()==orders;assert kw['name']=='eval-'+batch.group_id and kw['timeout'].to_seconds()==176400 and kw['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kw['max_retries_failure']==kw['max_retries_preemption']==0;assert kw['entrypoint'].workdir_files;fn,args,kwargs=kw['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kwargs);assert len(args)==1 and not kwargs and args[0].records_prefix==item['records_prefix']and args[0].model.location==target['location']and args[0].evaluations[0].identity.output_dir==batch.evaluations[0].identity.output_dir;assert kw['environment'].env_vars['MARIN_PREFIX']==item['MARIN_PREFIX'];r['SDK_EXACT_unredacted_MARIN_PREFIX']=kw['environment'].env_vars['MARIN_PREFIX'];assert'cw-rno2a'in json.dumps([MessageToDict(x.to_proto(),preserving_proto_field_name=True)for x in kw['constraints']]);r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
     try:job=client.submit(**kw)
     except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('ReadOnlyreconcile, no blindretry')from None
     r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected;log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save();return job
   m.submit(batch,Guard());assert r['application_submit_calls']==1
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];r['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_ms':q.timeout.milliseconds,'MARIN_PREFIX_redacted_RPC':q.environment.env_vars['MARIN_PREFIX'],'RPCredaction_noninjective_not_exactnamespaceproof':True};r['actual_native_binding_PASS']=q.bundle_id==ZIP and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000 and'cw-rno2a'in json.dumps(cs)and q.environment.env_vars['MARIN_PREFIX']==redact_value({'MARIN_PREFIX':item['MARIN_PREFIX']})['MARIN_PREFIX'];save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':r['job'],'arm':a.arm,'original_slot':a.run,'replacement':1,'actual_native_binding_PASS':r.get('actual_native_binding_PASS')}),flush=True);assert r.get('actual_native_binding_PASS')is True,'Submittedalready: reconcileonlybeforeothers, doNOTretry'
if __name__=='__main__':main()
