"""The TWO ordered A8 training/export roots; one guarded application per arm on combined CLEAR."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2,job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-a8-r1';C=Path('/home/karan/oa/marin-path-b-a8-on-share');N=P.parent/'t310/current-orders.md';PIN='4bfe87adbb03372d9b51ff4c87b8f3a637e54bfe'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
 f,k=filesystem_for(uri);return f.cat_file(k)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A8a','A8b'],required=True);ap.add_argument('--orders-sha256',required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);a=ap.parse_args();os.chdir(C);out=D/f'{a.arm}-owner-launch-margay-r1.json';assert not out.exists(),'Consumedjournal;readonlyreconcile'
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert sha(a.review.read_bytes())==a.review_sha256;review=json.loads(a.review.read_bytes());assert review['verdict']=='CLEAR_COMBINED_A8a_A8b_DATA_AND_NATIVE_LAUNCH_BINDINGS' and review['code_pin']==PIN and review['plan_sha256']=='f527fb76e294211e33e5e28eb1183bc6788ae5414f229542e047c410332e2a65' and review['review_receipt_sha256']=='bafddfe76c1afd5bc6ca6b52961fa9ab6c978c12959bc9ab2e310b1d408e547a' and review['native_bundle_sha256']=='2cdc175040f2788719cb5695c933a068ecaf199696c6d4bd10f72c3fa7ae051b' and review['staging_receipt_sha256']=='d6344552553507485c566e8bb4da2bf5d0e59301db674ee9d272ef67db156ecc'
 assert sha((D/'review-receipt.json').read_bytes())=='bafddfe76c1afd5bc6ca6b52961fa9ab6c978c12959bc9ab2e310b1d408e547a';assert sha((D/'launch-plan.json').read_bytes())=='f527fb76e294211e33e5e28eb1183bc6788ae5414f229542e047c410332e2a65';plan=json.loads((D/'launch-plan.json').read_bytes());assert plan['code_pin']==PIN and plan['workspace']==str(C) and len(plan['arms'])==2;target=next(t for t in plan['arms'] if t['arm']==a.arm);assert target['arm']==a.arm and target['checkpoint_updates']==[36] and not target['restore_from_checkpoint'] and target['frozen_native_tensors']==104 and target['trained_master_cast_tensors']==398 and target['special_rows_frozen'];assert plan['training']['updates']==36 and plan['training']['learning_rate']==2e-6
 for k in ('current_input_manifest','current_staging_receipt'):assert sha(Path(plan[k]['path']).read_bytes())==plan[k]['sha256']
 assert sha(Path(plan['runtime']['local']).read_bytes())==plan['runtime']['sha256']=='565cf0b2e09b0e1d58b647b698364a04d41b80615b47f8e6e1dbc7167f0d8e9e';stage=json.loads(Path(plan['current_staging_receipt']['path']).read_bytes());assert stage['state']=='ALL6_CPU_STAGE_FULLGET_PASS_NO_MODEL_CALL' and stage['staged_input_bytes']==9072506 and len(stage['stage_objects'])==6
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ('CW_KEY_ID','CW_KEY_SECRET')}
 for item in stage['stage_objects']:
  raw=get(item['uri']);assert len(raw)==item['bytes'] and sha(raw)==item['sha256']
 for k in ('source_receipt','source_objects'):assert sha(get(plan[k]['path']))==plan[k]['sha256']
 run_root=target['training_kwargs']['run_root'];fs,key=filesystem_for(run_root);assert not fs.exists(key) and not fs.find(key),'A8outputnotfresh'
 spec=importlib.util.spec_from_file_location('sealed_A8',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kwargs=mod.request(a.arm,credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);fn,args,call_kwargs=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**call_kwargs);assert not args;training=call_kwargs['training'];assert training['training_pin']==PIN and 'resume' not in training and training['runtime_bytes']==Path(plan['runtime']['local']).read_bytes();assert kwargs['name']==target['root_name'] and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400 and kwargs['resources'].to_proto().cpu_millicores==8000
 bundle=create_workspace_zip(C);bundle_sha=sha(bundle);assert bundle_sha=='2cdc175040f2788719cb5695c933a068ecaf199696c6d4bd10f72c3fa7ae051b'
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as client:
   orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'ON-share dose arms A8a / A8b' in orders;expected='/karan/'+target['root_name'];assert not client.list_jobs(prefix=expected)
   record={'state':'PRE_SUBMIT','utc':utc(),'arm':a.arm,'authority':'JerboaA8a/bGO+COMBINED6fc95936CLEAR;bothfreshrootsONCE/no diagnostic/identitypacketwait/no refill','code_pin':PIN,'plan_sha256':sha((D/'launch-plan.json').read_bytes()),'packet_sha256':sha((D/'review-receipt.json').read_bytes()),'review':str(a.review),'review_sha256':a.review_sha256,'staging_sha256':plan['current_staging_receipt']['sha256'],'orders_sha256_submission_only':sha(orders),'expected_job':expected,'run_root':run_root,'local_native_bundle_sha256':bundle_sha,'native_bundle_bytes':len(bundle),'application_submit_calls':0,'fresh_v01_no_resume':True,'completed_updates_requested':36,'only_final_native_step':35,'actual_dose':target['actual_dose'],'dose_comparison_reported_before_submit':True,'realized_final_replay_equal_A6_not_historical_A3':True,'per_update_padding_grouping_and_normalization_timing_diff_acknowledged':True,'selected_ON_occurrences':target['selected_ON_occurrences'],'padding_only_sequences':target['padding_only_sequences'],'frozen_native_tensors':104,'trained_master_cast_tensors':398,'special_rows_frozen':True,'CPU_probe_before_export_GPU':True,'HF_audit_before_native_retire_and_BFCL':True,'followup_scope':'auditedHFthenBFCL5+ORIGINALidentity132,NO P1-prefill/GSM/teacher/unseen','no_automatic_retry':True,'credentials_in_receipt':False}
   def save():
    with out.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with out.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   assert N.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Readonlyreconcile;neverautomaticallyretry') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();log=(D/f'{a.arm}-completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/f'{a.arm}-completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save()
   try:
    req=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request;record['actual_bundle_id']=req.bundle_id;record['actual_bundle_matches_local']=req.bundle_id==bundle_sha;save()
   except Exception as error:record['actual_bundle_readback_error_type']=type(error).__name__;save()
   print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'actual_bundle_matches_local':record.get('actual_bundle_matches_local'),'application_submit_calls':1}),flush=True)
if __name__=='__main__':main()
