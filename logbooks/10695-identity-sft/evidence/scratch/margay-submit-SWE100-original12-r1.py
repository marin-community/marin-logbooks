"""One original SWE100 logical slot; prepared now, blocked only on sealed narrow CLEAR."""
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
from botocore.session import get_session
from botocore.config import Config
from rigging.redaction import redact_value
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'second-agentic-SWE100-r1';E=P/'second-agentic-swe100-execution-r1';N=P.parent/'t310/current-orders.md'
REQUIRED_VERDICT='CLEAR_TWELVE_SWE100_EVALUATION_RUNS_ONLY'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['v01','A0','A3','A8a'],required=True);ap.add_argument('--run',choices=[1,2,3],type=int,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();assert sha(a.review.read_bytes())==a.review_sha256=='f514fa59aeda5f8f51f9fb18a94a7ed8ea96bea96638b84e30918a3867ca9248';review=json.loads(a.review.read_bytes());assert sha((D/'independent-binding-proof-marten-r1.json').read_bytes())==review['independent_binding_proof_sha256']=='1b3f8af7598b01eeb5466d110926a8732e212c9af37f347373241dcfbff0b51a';assert review['verdict']==REQUIRED_VERDICT,'Narrow execution CLEAR required, no other dependency';plan_raw=(D/'launch-plan.json').read_bytes();plan=json.loads(plan_raw);C=Path(plan['workspace']);os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==review['code_pin']==plan['code_pin'] and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert sha(plan_raw)==review['plan_sha256'];receipt_raw=(D/'review-receipt.json').read_bytes();assert sha(receipt_raw)==review['receipt_sha256']
 for path,row in json.loads(receipt_raw)['files'].items():
  raw=Path(path).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
 assert sha(Path(plan['suite']['path']).read_bytes())==plan['suite']['sha256']=='117cc38b33491a6a843793fb0f43314b7851c01f0da665ff70251fec33eed156';assert len(plan['suite']['task_ids'])==len(set(plan['suite']['task_ids']))==100
 all_runs=[r for t in plan['targets'].values()for r in t['runs']];assert len(all_runs)==12 and len({r['MARIN_PREFIX']for r in all_runs})==len({r['records_prefix']for r in all_runs})==12
 for key in ['preregistration','cost_choice']:assert sha(Path(plan[key]['path']).read_bytes())==plan[key]['sha256']
 slots=json.loads((E/'registered12-logical-slots-and-fixed100tasks-margay-r1.json').read_bytes());assert slots['fixed100_task_names']==plan['suite']['task_ids'];sd=E/a.arm/f'original-run-{a.run}';sd.mkdir(parents=True,exist_ok=True);out=sd/'launch.json';assert not out.exists(),'Consumed exclusive original-slot journal: reconcile read-only, no blind retry';target=plan['targets'][a.arm];item=next(r for r in target['runs'] if r['run']==a.run);assert sha(Path(target['model_config']).read_bytes())==target['model_config_sha256'];orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'SECOND AGENTIC EVAL' in orders
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 def get(uri):
  f,k=filesystem_for(uri);return f.cat_file(k)
 guards=target['guards'];keys=['source_receipt','source_objects'] if guards['kind']=='published_source' else ['event','audit']
 for name in keys:assert sha(get(guards[name]['uri']))==guards[name]['sha256']
 for uri in [item['records_prefix'],item['MARIN_PREFIX']]:
  f,k=filesystem_for(uri);assert not f.exists(k) and not f.find(k),'Fresh namespace required'
 bundle=create_workspace_zip(C);bundle_sha=sha(bundle);assert bundle_sha==review['native_bundle_sha256'];spec=importlib.util.spec_from_file_location('sealed_SWE100_execution',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);batch=mod.prepare(a.arm,a.run);assert batch.model.location==target['location'] and len(batch.evaluations)==1 and batch.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE;assert os.environ['MARIN_PREFIX']==item['MARIN_PREFIX'] and batch.records_prefix==item['records_prefix'];assert batch.evaluations[0].identity.output_dir==item['records_prefix'].rstrip('/')+'/'+batch.evaluations[0].identity.run_id+'/results','Wrong native records/run_id/results namespace';expected='/karan/eval-'+batch.group_id
 record={'state':'PRE_SUBMIT','utc':utc(),'authority':'Jerboa SWE100 fixed100 mini2.1 12originals + narrowbindingCLEAR','arm':a.arm,'original_slot':a.run,'source_pin':plan['code_pin'],'plan_sha256':sha(plan_raw),'packet_sha256':sha(receipt_raw),'review_sha256':a.review_sha256,'orders_sha256_submission_only':sha(orders),'local_native_bundle_sha256':bundle_sha,'model_yaml_sha256':target['model_config_sha256'],'model_HF':target['location'],'records_prefix':item['records_prefix'],'MARIN_PREFIX':item['MARIN_PREFIX'],'expected_job':expected,'group_id':batch.group_id,'registered_native_evaluations':[{'run_id':e.identity.run_id,'eval_name':e.identity.eval_ref.name,'actual_prepared_output_dir':e.identity.output_dir}for e in batch.evaluations],'all100_original_task_names_frozen_BEFORE_any_transcript_read':plan['suite']['task_ids'],'scientific_config_sha256':plan['suite']['sha256'],'standard_Harbor_internal_infra_retry_policy_unchanged':plan['policy']['internal_Harbor_retry'],'sampling_agent_parser_renderer_changes':False,'application_submit_calls':0,'no_automatic_retry_uncertain_submit':True,'no_A9_A10_KEEP_or_baseline_dependency':True,'native_checkpoint_ops':0,'credentials_or_environment_values_in_receipt':False}
 def save():
  with out.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with out.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 probe=secrets.token_bytes(32);probe_key='tmp/ttl=14d/snowball-identity/path-b/SWE100-submit-probes-43bba4a16d17/'+a.arm+'-'+str(a.run)+'-'+secrets.token_hex(8)+'.bin';probe_uri='s3://marin-rno2a/'+probe_key;record['write_probe']={'uri':probe_uri,'bytes':32,'sha256':sha(probe),'state':'PUT_CALLED','transport':'Synchronous botocore CPU preflight only, connect10/read20s, one transport attempt; no scientific request change'};save();probe_client=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=resolve_secret_spec('env:CW_KEY_ID').value,aws_secret_access_key=resolve_secret_spec('env:CW_KEY_SECRET').value,config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
 try:
  probe_client.put_object(Bucket='marin-rno2a',Key=probe_key,Body=probe);response=probe_client.get_object(Bucket='marin-rno2a',Key=probe_key);body=response['Body']
  try:assert body.read()==probe
  finally:body.close()
 finally:probe_client.close()
 record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as client:
   assert not client.list_jobs(prefix=expected)
   class GuardedClient:
    def submit(self,**kwargs):
     assert record['application_submit_calls']==0 and N.read_bytes()==orders and sha(a.review.read_bytes())==a.review_sha256;assert kwargs['name']=='eval-'+batch.group_id and kwargs['timeout'].to_seconds()==176400 and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;assert kwargs['entrypoint'].workdir_files,'Preserve native callable serialization';fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert len(args)==1 and not kw and args[0].records_prefix==item['records_prefix'] and args[0].model.location==target['location'] and args[0].evaluations[0].identity.output_dir==item['records_prefix'].rstrip('/')+'/'+args[0].evaluations[0].identity.run_id+'/results';assert kwargs['environment'].env_vars['MARIN_PREFIX']==item['MARIN_PREFIX'],'Actual root environment belongs to wrong run';record['root_env_MARIN_PREFIX_checked_at_actual_call']=kwargs['environment'].env_vars['MARIN_PREFIX'];constraints=[MessageToDict(x.to_proto(),preserving_proto_field_name=True)for x in kwargs['constraints']];assert 'cw-rno2a' in json.dumps(constraints);record.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc(),required_serialization_preserved=True);save()
     try:job=client.submit(**kwargs)
     except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Read-only reconcile; no blind retry')from None
     record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expected;log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();return job
   submitted=mod.submit(batch,GuardedClient());assert record['application_submit_calls']==1
   try:
    native=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job']));q=native.request;record['actual_native_request']={'bundle_id':q.bundle_id,'constraints':[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints],'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'priority':job_pb2.PriorityBand.Name(q.priority_band),'timeout_ms':q.timeout.milliseconds,'actual_root_env_MARIN_PREFIX_RPC_REDACTED':q.environment.env_vars['MARIN_PREFIX'],'namespace_exact_at_SDK_call_not_claimed_unredacted_RPC':True};record['actual_native_binding_PASS']=q.bundle_id==bundle_sha and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000 and q.environment.env_vars['MARIN_PREFIX']==redact_value({'MARIN_PREFIX':item['MARIN_PREFIX']})['MARIN_PREFIX'] and 'cw-rno2a' in json.dumps(record['actual_native_request']['constraints']);save()
   except Exception as error:record['native_readback_error_type']=type(error).__name__;save()
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'arm':a.arm,'original_slot':a.run,'actual_native_binding_PASS':record.get('actual_native_binding_PASS')}),flush=True)
 assert record.get('actual_native_binding_PASS') is True,'Already submitted: reconcile read-only before other slots; never retry this consumed slot'
if __name__=='__main__':main()
