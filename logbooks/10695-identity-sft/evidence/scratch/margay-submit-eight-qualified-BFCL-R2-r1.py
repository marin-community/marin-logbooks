"""ONE of eight authorized whole-BFCL R2s; namespace-only successor of a sealed R2 packet. No R3."""
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
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');N=P.parent/'t310/current-orders.md';C=Path('/home/karan/oa/marin-path-b-swe100')
PIN='43bba4a16d175072b20fea27b8fb5524c377c79f';ZIP='5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b'
HELPER='a604c7dfbfb1b36b552fad4605199580bfed43ae29f4e119c04de60c6b1e3af3';CENSUS='13d4eb22516327d6c0661d55548d1fc851b1365608a1636834dd49dae9835449'
PACKETS={'A13':('A13-BFCL-infra-replacement2-slots4-r1','961511dbdcd36adda35f23636f5686442e338b431d040b156acfff11c4447c11',[4]),
 'M25':('M25-BFCL-infra-replacement2-slots4-r1','336a55345d9fbb87223484645ddbd28136977482b8af28102d5157fceefbdb6f',[4]),
 'M49':('M49-BFCL-infra-replacement2-slots1-2-r1','0ca29f35393209c0fd06904eca7a7a490a917a5498fbab6206b7ee5f47dbafa3',[1,2]),
 'M51':('M51-BFCL-infra-replacement2-slots1-r1','63fae3895852e89e537850fe7e433090859a8b9027ee2ca09a2a688d97fa2ec6',[1]),
 'M75':('M75-BFCL-infra-replacement2-slots1-2-3-r1','2f6ac3022c71d31800ccc94a2df778ddac576d6d0053a9ac71fbec048d6f3525',[1,2,3])}
PROD=job_pb2.PRIORITY_BAND_PRODUCTION
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=list(PACKETS),required=True);ap.add_argument('--slot',type=int,choices=range(1,6),required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C)
 folder,receipt_sha,slots=PACKETS[a.arm];assert a.slot in slots;D=P/folder;assert sha((D/'review-receipt.json').read_bytes())==receipt_sha
 for path,x in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==x['bytes']and sha(b)==x['sha256'],path
 assert sha((D/'prepare_request.py').read_bytes())==HELPER
 plan=json.loads((D/'launch-plan.json').read_bytes());assert plan['code_pin']==PIN and plan['priority']=='production' and plan['replacement_number']==2 and plan['NO_R3'];assert plan['qualification_aggregate']['sha256']==CENSUS and sha(Path(plan['qualification_aggregate']['path']).read_bytes())==CENSUS
 target=plan['targets'][a.arm];item=next(x for x in target['runs']if x['run']==a.slot);assert item['replacement_number']==2
 assert item['version'] not in{item['original_configuration']['version'],item['replacement1_configuration']['version']}
 for key in['records_prefix','MARIN_PREFIX']:
  assert item[key]not in{item['original_configuration'][key],item['replacement1_configuration'][key]}
 assert item['records_prefix'].endswith(f'run-{a.slot}-infra-replacement-2/records')and item['MARIN_PREFIX'].endswith(f'run-{a.slot}-infra-replacement-2/native')
 chain=next(x for x in plan['qualified_chains']if x['original_slot']==a.slot);assert chain['primary_score']is None and 'NOT_EXHAUSTED' in chain['state'];physical={x['replacement_number']:x for x in chain['physical']};assert set(physical)=={0,1}
 for number,x in physical.items():
  b=Path(x['proof']).read_bytes();assert sha(b)==x['proof_sha256'];r=json.loads(b);assert r['arm']==a.arm and r['original_slot']==a.slot and r['replacement_number']==number and not r['logical_full123_admissible']and r['score']is None and r['HF']==target['location'];unscored=[e for e in r['native_scored_or_unscored_errors_preserved']if not e['scored']];assert unscored and all(e['reward']is None and e['error']['category']=='infrastructure'for e in unscored);assert all(e['reward']in[0,1]for e in r['native_scored_or_unscored_errors_preserved']if e['scored']);assert r['owned_terminal_attempts_and_scoped_cost']['all_owned_jobs_terminal'];assert len(r['rows'])==123
 census=json.loads(Path(plan['qualification_aggregate']['path']).read_bytes());cent=next(x for x in census['pending_BFCL_logical_chains']if x['arm']==a.arm and x['original_slot']==a.slot);assert{x['replacement_number']:x['proof_sha256']for x in cent['physical']}=={n:x['proof_sha256']for n,x in physical.items()}
 assert sha(Path(plan['suite']['path']).read_bytes())==plan['suite']['sha256']=='73bf0dc40d438b04fedaf1ac31bc4c35d3fc07ce851d815730c5079f168666e1';readyref=plan['actual_readiness'];rb=Path(readyref['path']).read_bytes();assert sha(rb)==readyref['sha256'];ready=json.loads(rb);assert ready['HF']==target['location']
 sb=Path(plan['original_slot1_cohort']['path']).read_bytes();assert sha(sb)==plan['original_slot1_cohort']['sha256'];sd=D/'BFCL-whole-replacement2'/f'run-{a.slot}';sd.mkdir(parents=True,exist_ok=True);out=sd/'launch.json';assert not out.exists(),'R2 already consumed; reconcile read-only'
 orders=N.read_bytes();assert sha(orders)==a.orders_sha256
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 for x in ready['metadata'].values():
  f,k=filesystem_for(x['uri']);b=f.cat_file(k);assert len(b)==x['bytes']and sha(b)==x['sha256']
 for uri in[item['records_prefix'],item['MARIN_PREFIX']]:
  f,k=filesystem_for(uri);assert not f.exists(k)and not f.find(k)
 spec=importlib.util.spec_from_file_location('sealed_R2',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);batch=m.prepare(a.arm,a.slot);assert batch.model.location==target['location']and batch.records_prefix==item['records_prefix']and batch.priority_band==PROD and os.environ['MARIN_PREFIX']==item['MARIN_PREFIX']and len(batch.evaluations)==1;ev=batch.evaluations[0];assert ev.identity.output_dir==batch.records_prefix+'/'+ev.identity.run_id+'/results'
 def owned(b):
  cfg=m.held_bfcl.inference_config_for_model(b.model,b.accelerator,env_vars={},priority=b.priority_band,capability_origin=b.capability_origin,api_model=b.api_model);assert cfg.iris.priority==PROD
 owned(batch);jobname='/karan/eval-'+batch.group_id
 r={'utc':utc(),'state':'PRE_PROBE','arm':a.arm,'original_slot':a.slot,'replacement_number':2,'authority':'Director R2 order + standing infra rule: ONE whole fresh unchanged R2 per infra-incomplete original AND R1; NO R3; namespace-only change','original_job':physical[0]['root_job'],'r1_job':physical[1]['root_job'],'original_proof_sha256':physical[0]['proof_sha256'],'r1_proof_sha256':physical[1]['proof_sha256'],'qualification_aggregate_sha256':CENSUS,'census_sha256':CENSUS,'r2_packet_receipt_sha256':receipt_sha,'prepare_helper_sha256':HELPER,'plan_sha256':sha((D/'launch-plan.json').read_bytes()),'source_pin':PIN,'local_native_workspace_bundle_sha256':ZIP,'orders_sha256_submission_only':sha(orders),'expected_job':jobname,'HF':target['location'],'HF_readiness_proof_sha256':sha(rb),'records_prefix':item['records_prefix'],'MARIN_PREFIX':item['MARIN_PREFIX'],'r2_version':item['version'],'registered_native_evaluations':[{'run_id':ev.identity.run_id,'output_dir':ev.identity.output_dir}],'all123_preserved_no_task_repair_or_zero_imputation':True,'application_submit_calls':0,'offered_root_AND_owned_serve_PRODUCTION':True,'global_child_mirror_NOT_original_peer_policy_or_priority_attestation':True,'NO_R3':True,'credentials_in_receipt':False}
 def save():
  with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n')
 body=secrets.token_bytes(32);key=f'tmp/ttl=14d/snowball-identity/path-b/qualified-BFCL-R2-submit-probes/{a.arm}-{a.slot}-'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'};save()
 s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=resolve_secret_spec('env:CW_KEY_ID').value,aws_secret_access_key=resolve_secret_spec('env:CW_KEY_SECRET').value,config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
 try:
  s3.put_object(Bucket='marin-rno2a',Key=key,Body=body);stream=s3.get_object(Bucket='marin-rno2a',Key=key)['Body']
  try:assert stream.read()==body
  finally:stream.close()
 finally:s3.close()
 r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=jobname)
   class Guard:
    def submit(self,**kwargs):
     assert r['application_submit_calls']==0 and N.read_bytes()==orders and sha(Path(plan['qualification_aggregate']['path']).read_bytes())==CENSUS and sha((D/'review-receipt.json').read_bytes())==receipt_sha
     assert kwargs['name']=='eval-'+batch.group_id and kwargs['priority_band']==PROD and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400 and kwargs['entrypoint'].workdir_files;fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert len(args)==1 and not kw;actual=args[0];assert actual.model.location==target['location']and actual.records_prefix==item['records_prefix']and actual.priority_band==PROD and actual.evaluations[0].identity.output_dir==ev.identity.output_dir;owned(actual);assert kwargs['environment'].env_vars['MARIN_PREFIX']==item['MARIN_PREFIX'];assert'cw-rno2a'in json.dumps([MessageToDict(c.to_proto(),preserving_proto_field_name=True)for c in kwargs['constraints']]);r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc(),root_exact_SDK_namespace_and_owned_producer_PRIORITY_verified=True);save()
     try:job=client.submit(**kwargs)
     except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Read-only reconcile, never blind retry')from None
     r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==jobname;log=(sd/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(sd/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save();return job
   m.submit(batch,Guard());assert r['application_submit_calls']==1
   qreq=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(c,preserving_proto_field_name=True)for c in qreq.constraints]
   r['actual_root_stored_request']={'bundle_id':qreq.bundle_id,'priority':job_pb2.PriorityBand.Name(qreq.priority_band),'timeout_ms':qreq.timeout.milliseconds,'failure_retries':qreq.max_retries_failure,'preemption_retries':qreq.max_retries_preemption,'constraints':cs}
   r['actual_native_binding_PASS']=qreq.bundle_id==ZIP and qreq.priority_band==PROD and qreq.timeout.milliseconds==176400000 and qreq.max_retries_failure==qreq.max_retries_preemption==0 and'cw-rno2a'in json.dumps(cs);save()
 print('QUALIFIED_WHOLE_BFCL_R2_ACTUALLY_SUBMITTED',a.arm,a.slot,str(out),sha(out.read_bytes()),r['job'],'NATIVE_PASS',r['actual_native_binding_PASS'],flush=True);assert r['actual_native_binding_PASS'],'Already submitted; do not repeat'
if __name__=='__main__':main()
