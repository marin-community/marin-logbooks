"""ONE authorized A13-only private HF archive (step 3); no S3 deletion, no model evaluation.
Executor-side override: the sealed hf_archive.submission() hardcodes the r2 job name, so this sets a fresh
r3 name rather than editing the sealed workspace."""
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
from experiments.post_training.identity_path_b.hf_archive import ARTIFACTS,WARNING,RESPONSIBLE_USE
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'private-HF-archive-r3-A13';C=Path('/home/karan/oa/marin-path-b-hf-archive-r2');N=P.parent/'t310/current-orders.md'
PIN='cd8348baef2419b8a8ef305fa8308933690393a3';R='b949db5574eecbf262067190ecbe8766dbfb26369a397ea902592326a191dfd5';PL='c3dfbf4a7e502abef173d542709f746d496beffbddf0b8916e65bb61f6811819';ZIP='784fd920786ccd9b58e41ab5f8484d50b5d22af1c4ec99529fe1b91255321326'
NEWNAME='pathb-private-HF-archive-r3-A13-20261031'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);out=D/'actual-CPU-upload-launch-margay-r1.json';assert not out.exists(),'Consumed journal: read-only reconcile, never automatic retry'
 assert sha((D/'receipt.json').read_bytes())==R and sha((D/'launch-plan.json').read_bytes())==PL
 plan=json.loads((D/'launch-plan.json').read_bytes())
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 for x in json.loads((D/'receipt.json').read_bytes())['files']:
  b=Path(x['path']).read_bytes();assert len(b)==x['bytes']and sha(b)==x['sha256'],x['path']
 for k,v in plan['source_copy_evidence'].items():
  if isinstance(v,dict)and 'path'in v:assert sha(Path(v['path']).read_bytes())==v['sha256'],k
 assert plan['source_copy_evidence']['copy_spec']['sha256']=='d1e71b3059ed5f4102ac6821d0b5b61db72d0513f48ef9e249e72f018ac544ee'
 assert plan['source_copy_evidence']['launch']['sha256']=='cc903d5593f0dfa8203cbde76562ad731083828ae39390d39abbde8912c143e2'
 assert not plan['deletion']['performed_by_job']and not plan['deletion']['delete_eligible']
 assert sha(Path(plan['prior_packet']['path']).read_bytes())==plan['prior_packet']['sha256']=='1ade5cab29239fda93ad61a4934f6b52ce9b3ea8703f4ee19a8dda16458bc3a5'
 assert sha((ARTIFACTS/'LICENSE.OpenMDW-1.1').read_bytes())==plan['official_license']['sha256']=='2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df'
 orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'HF ARCHIVE' in orders
 for target in plan['targets']:
  text=(ARTIFACTS/target['card_file']).read_text();assert text.count(WARNING)==text.count(RESPONSIBLE_USE)==1 and 'license: openmdw-1.1' in text
  assert target['repo_id']=='open-athena/Snowball-67B-A2B-Identity-PathB-'+target['label']+'-2026.10.30'
  assert sha((ARTIFACTS/target['manifest_file']).read_bytes())==target['manifest_sha256']and sha((ARTIFACTS/target['keep_file']).read_bytes())==target['keep_sha256']
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={key:require_secret(key)for key in ['CW_KEY_ID','CW_KEY_SECRET','HF_TOKEN']}
 spec=importlib.util.spec_from_file_location('sealed_private_HF_archive_r3',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);kwargs=m.request(credentials)
 assert kwargs['name']=='pathb-private-HF-archive-r2-20261030','sealed submission() name changed; re-check the override'
 kwargs['name']=NEWNAME   # executor-side override: avoid colliding with the completed r2 job of the same name
 inspect.signature(IrisClient.submit).bind(object(),**kwargs);assert kwargs['entrypoint'].workdir_files;fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert not args and not kw
 assert kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==43200
 assert set(kwargs['environment'].secrets)=={'CW_KEY_ID','CW_KEY_SECRET','HF_TOKEN'}and kwargs['environment'].env_vars is None
 constraints=[MessageToDict(x.to_proto(),preserving_proto_field_name=True)for x in kwargs['constraints']];assert'cw-rno2a'in json.dumps(constraints)
 assert sha(create_workspace_zip(C))==ZIP,'native bundle is not the sealed 784fd920'
 fs,k=filesystem_for(plan['report_root']);assert not fs.exists(k)and not fs.find(k),'Report namespace not fresh'
 expected='/karan/'+kwargs['name']
 r={'utc':utc(),'state':'PRE_SUBMIT','authority':'Jerboa-approved vizcacha 3-step A13 archive route, step 3 (A13-only upload) with the unchanged worker; explicit PRIVATE open-athena upload; no science review gate','step':3,'code_pin':PIN,'receipt_sha256':R,'plan_sha256':PL,'orders_sha256_submission_only':sha(orders),'expected_job':expected,'executor_side_job_name_override':{'sealed_name':'pathb-private-HF-archive-r2-20261030','used_name':NEWNAME,'why':'the sealed hf_archive.submission() hardcodes the r2 job name, which already exists as a SUCCEEDED job; reusing it would collide and be ambiguous. No workspace edit was made.'},'native_bundle_sha256':ZIP,'requested_resources':plan['resources'],'private_targets':plan['targets'],'report_root':plan['report_root'],'source_checks':[],'original_native_callable_workdir_files_preserved':True,'secrets_names_only':sorted(credentials),'credentials_in_receipt':False,'all56sourcefiles_per_model_worker_FULLSTREAM_hash_then_allHub_file_hash_verification_REQUIRED':True,'NO_S3_deletion':True,'delete_eligible':False,'application_submit_calls':0,'automatic_retry':False}
 s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=30,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
 def get(key):
  resp=s3.get_object(Bucket='marin-rno2a',Key=key);stream=resp['Body']
  try:return stream.read()
  finally:stream.close()
 def save():
  with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 try:
  for target in plan['targets']:
   prefix=target['source'].removeprefix('s3://marin-rno2a/').rstrip('/')+'/'
   manifest=json.loads((ARTIFACTS/target['manifest_file']).read_bytes())
   listing={x['Key'][len(prefix):]:x for page in s3.get_paginator('list_objects_v2').paginate(Bucket='marin-rno2a',Prefix=prefix)for x in page.get('Contents',[])}
   assert set(listing)=={x['relative']for x in manifest['objects']}|{'KEEP.md','PRESERVATION-SHA256.json'},'source inventory != manifest objects + the two documents'
   for x in manifest['objects']:assert listing[x['relative']]['Size']==x['bytes']and listing[x['relative']]['ETag']==x['destination_etag']
   for key,local in[('PRESERVATION-SHA256.json',target['manifest_file']),('KEEP.md',target['keep_file'])]:assert get(prefix+key)==(ARTIFACTS/local).read_bytes(),key
   audit_raw=get(prefix+'path-b-storage-audit.json');assert sha(audit_raw)==target['audit_sha256']
   audit=json.loads(audit_raw);assert len(audit['tensors'])==502 and audit['frozen_native_tensors']==104 and audit['trained_master_cast_tensors']==398
   assert sha(get(prefix+'path-b-checkpoint-event.json'))==target['event_sha256']
   index=json.loads(get(prefix+'model.safetensors.index.json'));assert len(set(index['weight_map'].values()))==39 and set(index['weight_map'].values())<=set(listing)
   r['source_checks'].append({'arm':target['label'],'exact58_object_inventory':True,'all56_size_ETag_unchanged_from_the_FULLSTREAM_SHA_copy_proof':True,'manifest_KEEP_event_audit_index_FULLGET_verified':True,'indexed_shards_present':39,'weight_FULLSTREAM_reverification_by_worker_not_operator_host':True})
  with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
  probe=secrets.token_bytes(32);key='tmp/ttl=14d/snowball-identity/path-b/private-HF-archive-submit-probes/'+secrets.token_hex(8)+'.bin';r['write_probe']={'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(probe),'state':'PUT_CALLED'};save();s3.put_object(Bucket='marin-rno2a',Key=key,Body=probe);assert get(key)==probe;r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save()
 finally:s3.close()
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected),'r3 job name not fresh'
   assert N.read_bytes()==orders and sha((D/'receipt.json').read_bytes())==R
   r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Read-only reconcile, no automatic retry')from None
   r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected
   log=(D/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','45000','--out',str(D/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save()
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request;cs=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints]
    r['actual_native_request']={'bundle_id':q.bundle_id,'constraints':cs,'priority':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_ms':q.timeout.milliseconds,'resource_spec':MessageToDict(q.resources,preserving_proto_field_name=True)}
    r['actual_native_binding_PASS']=q.bundle_id==ZIP and'cw-rno2a'in json.dumps(cs)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==43200000;save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':r['job'],'actual_native_binding_PASS':r.get('actual_native_binding_PASS'),'one_notifier_pid':proc.pid,'delete_eligible':False}),flush=True)
if __name__=='__main__':main()
