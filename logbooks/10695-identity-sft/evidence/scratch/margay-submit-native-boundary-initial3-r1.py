"""One cleared native-boundary132 per initial target. No model submit before actual write-gate PASS."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2,job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
from experiments.post_training.identity_path_b.native_boundary_scope.owner import checked_identity
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'native-boundary-identity132-r1';C=Path('/home/karan/oa/marin-path-b-native-boundary132');N=P.parent/'t310/current-orders.md';PIN='62550b5c905e63e57121576edf8806419e76d0bb';ZIP='726ef4e2b205ca0376f29ec06484d05c6ab7e8b67febc8c634df44e48dac26c7';CLEAR='d4c5e457a234b3c48a554ec73de98d349521f8856c50fb8806c768e273937598'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['v01','A3','A8a'],required=True);ap.add_argument('--orders-sha256',required=True);ap.add_argument('--write-gate-receipt',type=Path,required=True);ap.add_argument('--write-gate-sha256',required=True);a=ap.parse_args();os.chdir(C);out=D/a.target/'actual-launch-margay-r1.json';out.parent.mkdir(exist_ok=True);assert not out.exists(),'Exclusive ONCE journal exists; read-only reconcile, no blind retry';gb=a.write_gate_receipt.read_bytes();assert sha(gb)==a.write_gate_sha256;gate=json.loads(gb);assert gate['state']=='PUT_FULLGET_32B_BYTE_EQUAL_PASS';assert 'marin-rno2a' in json.dumps(gate);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'native-boundary identity-132' in orders
 expected={'review-receipt.json':'3b0f47186564cc99c3bb8c2cce1f9a95265492db40d7fd54e03c5351077809cb','launch-plan.json':'ef913d3495d0992bdfae7d18f38bd82c1302667043a1890ddf0d17ce46b35696','science-verdict-marten-r1.json':CLEAR}
 for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
 plan=json.loads((D/'launch-plan.json').read_bytes());v=json.loads((D/'science-verdict-marten-r1.json').read_bytes());assert v['verdict']=='CLEAR_THREE_NATIVE_BOUNDARY_IDENTITY132_SCREENS_ONLY' and v['code_pin']==PIN and v['native_bundle_sha256']==ZIP;assert sha(Path(plan['preregistration']['path']).read_bytes())==plan['preregistration']['sha256'];item=plan['targets'][a.target];expected_job='/karan/'+item['root_name'];load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k)for k in ['CW_KEY_ID','CW_KEY_SECRET']};actual=checked_identity(a.target,item['root_name']);spec=importlib.util.spec_from_file_location('sealed_native_boundary_initial3',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);kwargs=m.request(a.target,credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);assert kwargs['name']==item['root_name']and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400;assert not kwargs['entrypoint'].workdir_files,'Command-based diagnostic has no callable customfile dependency';assert sha(create_workspace_zip(C))==ZIP;f,k=filesystem_for(item['output']);assert not f.exists(k)and not f.find(k),'Output exists: no rerun/overwrite'
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected_job);r={'utc':utc(),'state':'PRE_SUBMIT_AFTER_ACTUAL_WRITE_GATE','target':a.target,'expected_job':expected_job,'output':item['output'],'code_pin':PIN,'native_bundle_sha256':ZIP,'clear_sha256':CLEAR,'orders_sha256_submission_only':sha(orders),'write_gate_receipt':str(a.write_gate_receipt),'write_gate_sha256':sha(gb),'actual_source_HF_metadata_FULLGET':actual,'request_delta_ONLY':{'skip_special_tokens':False,'return_token_ids':True},'no_P1_no_added_START_no_identity_system':True,'generatedIDs_authoritative_FIRSTgenerated128003_noEndNoONfinalCredit':True,'all_raw_HTTP_provider_fields_usage_finish_IDS_preserved':True,'application_submit_calls':0,'automatic_retry':False,'credentials_in_receipt':False}
   def save():
    with out.open('w')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with out.open('x')as h:json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   assert N.read_bytes()==orders
   for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
   r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as e:r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Read-only reconciliation ONLY, no blind retry')from None
   r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected_job;log=(out.parent/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(out.parent/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save()
   try:
    raw=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job']));q=raw.request;constraints=[MessageToDict(x,preserving_proto_field_name=True)for x in q.constraints];r['actual_native_request']={'bundle_id':q.bundle_id,'constraints':constraints,'priority_band':job_pb2.PriorityBand.Name(q.priority_band),'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption,'timeout_milliseconds':q.timeout.milliseconds,'resources':MessageToDict(q.resources,preserving_proto_field_name=True)};r['actual_native_binding_PASS']=q.bundle_id==ZIP and 'cw-rno2a'in json.dumps(constraints)and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000;save()
   except Exception as e:r['native_readback_error_type']=type(e).__name__;save()
   print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':r['job'],'one_notifier_pid':proc.pid,'actual_native_binding_PASS':r.get('actual_native_binding_PASS')}),flush=True)
if __name__=='__main__':main()
