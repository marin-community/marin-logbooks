"""ONE Jerboa-authorized A1 export-only recovery; never retrain or resubmit."""
import datetime,hashlib,inspect,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2,job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from experiments.post_training.identity_path_b.checkpoints import event_bytes
N=Path('/home/karan/Dropbox/marin-notes/snowball-identity');D=N/'path-b/bfcl-ablation-A1-export-recovery-r1';C=Path('/home/karan/oa/marin-path-b-ablation-recovery');PIN='278502ec5cee8b729b18bd0fa59b23cfa068e3fb';ORIGINAL='6ac2c53d061156ad22be53b332b7bd677a856290';OUT=D/'owner-launch-margay-r1.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
 f,k=filesystem_for(uri);return f.cat_file(k)
def main():
 os.chdir(C);assert not OUT.exists(),'Consumed one-shot recovery journal; reconcile only';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();raw=(D/'receipt.json').read_bytes();assert sha(raw)=='5f70a4e0f1df71d88fb77b9713619a8ea06ece97136e9b9e87a73f3f99a3c46d';receipt=json.loads(raw);assert len(receipt['files'])==26
 for b in receipt['files']:
  payload=Path(b['path']).read_bytes();assert len(payload)==b['bytes'] and sha(payload)==b['sha256'],b['path']
 plan_raw=(D/'launch-plan.json').read_bytes();assert sha(plan_raw)==receipt['plan_sha256']=='2c3d379ca8599d5111ebaa7ce41a262e20cf801ef4c75cda9a4fdb30b75afe41';plan=json.loads(plan_raw)
 changed=subprocess.check_output(['git','diff','--name-only',ORIGINAL,'HEAD'],text=True).splitlines();assert set(changed)=={'experiments/post_training/identity_path_b/ablation_recovery.py','experiments/post_training/identity_path_b/test_ablation_recovery.py'}
 for path in ('experiments/post_training/identity_path_b/ablation.py','experiments/post_training/identity_path_b/ablation_launch.py'):
  assert (C/path).read_bytes()==subprocess.check_output(['git','show',ORIGINAL+':'+path]),'Exporter/retirement changed'
 orders_path=N/'t310/current-orders.md';orders=orders_path.read_bytes();assert sha(orders)=='841dce13561069e50b605b1b3119952d2cc948dfa4e36fc71cad62319c35e489';load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:os.environ[k] for k in ('CW_KEY_ID','CW_KEY_SECRET')};root=plan['training_kwargs']['run_root'];evidence=plan['evidence']
 for uri,h in [(root+'/training-result.json',evidence['training_result_sha256']),(root+'/final-checkpoint-event.json',evidence['event_sha256']),(plan['event']['native_checkpoint']+'/metadata.json',evidence['metadata_sha256'])]:assert sha(get(uri))==h
 hf=plan['event']['hf_export'];f,k=filesystem_for(hf);objects={p:v for p,v in f.find(k,detail=True).items() if v.get('type')!='directory'};source_key=k.rstrip('/')+'/admission/audits/source-bind.json';assert set(objects)=={source_key} and objects[source_key]['size']==641;source=get(hf+'/admission/audits/source-bind.json');assert len(source)==evidence['source_bind_bytes']==641 and sha(source)==evidence['source_bind_sha256'];f,k=filesystem_for(plan['recovery_root']);assert not f.exists(k) and not f.find(k),'Recoveryoutputnotfresh';sys.path.insert(0,str(D));from prepare_request import request
 spec=request(credentials);inspect.signature(IrisClient.submit).bind(object(),**spec);fn,args,kwargs=spec['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kwargs);assert fn.__name__=='export_only_worker' and not args and set(kwargs)=={'training','event','recovery_root','evidence'};assert sha(event_bytes(kwargs['event']))==evidence['event_sha256'] and kwargs['training']['training_pin']==ORIGINAL and kwargs['recovery_root']==plan['recovery_root'] and kwargs['evidence']==evidence;assert spec['name']==plan['root_name'] and spec['timeout'].to_seconds()==176400 and spec['resources'].to_proto().cpu_millicores==8000 and spec['max_retries_failure']==spec['max_retries_preemption']==0 and spec['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE;bundle=create_workspace_zip(C);bundle_sha=sha(bundle)
 record={'state':'PRE_SUBMIT','utc':utc(),'authority':'Jerboa direct ONE A1 export-only recovery after queue preemption; no retrain, anysecondrecoveryneedsnewauthority','recovery_code_pin':PIN,'original_training_pin':ORIGINAL,'packet_receipt_sha256':sha(raw),'plan_sha256':sha(plan_raw),'original_event_sha256':evidence['event_sha256'],'native_metadata_sha256':evidence['metadata_sha256'],'original_source_bind_sha256':sha(source),'only_641B_source_bind_present_no_model_parts':True,'native_checkpoint':plan['event']['native_checkpoint'],'hf_export':hf,'recovery_root':plan['recovery_root'],'expected_job':'/karan/'+spec['name'],'orders_sha256_submission_only':sha(orders),'local_native_bundle_sha256':bundle_sha,'application_submit_calls':0,'training_job_submit_calls':0,'no_automatic_retry':True,'credentials_never_in_argv_receipts_or_env_dumps':True,'CPU_owner_timeout_seconds':176400,'GPU_export_requested_hosts':1,'GPU_export_requested_count':8,'GPU_export_timeout_seconds':7200,'new_probe_before_export_GPU':True,'source_bind_archive_full_readback_before_GPU':True,'native_delete_only_after_terminal_502audit_and_unchanged_receipt':True}
 def save():
  with OUT.open('w') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as c:
   assert not c.list_jobs(prefix=record['expected_job'])
   with OUT.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   assert orders_path.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=c.submit(**spec)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile exactnamespace readonly, NEVER resubmit') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==record['expected_job'];save();req=c._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request;record.update(actual_bundle_id=req.bundle_id,actual_bundle_matches_local=req.bundle_id==bundle_sha);save();assert record['actual_bundle_matches_local']
 print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),**record}),flush=True)
if __name__=='__main__':main()
