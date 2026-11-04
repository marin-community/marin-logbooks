"""One fresh36 A10 root on exact combined CLEAR; no other workload dependency."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,secrets,subprocess,sys
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
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-a10-r1';C=Path('/home/karan/oa/marin-path-b-a10-tulu');N=P.parent/'t310/current-orders.md';PIN='cf4da6f40326df150eac712e2eb323667f71a267';BUNDLE='ef68c3012fdb7911f8f47315236722fb5125c6f86ea72bd264abf7f927af6a18';OUT=D/'actual-training-export-launch-margay-r1.json'
REQUIRED_VERDICT='CLEAR_ONE_FRESH_A10_ARM_AND_STANDING_AUDITED_HF_FOLLOWUPS'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
 f,k=filesystem_for(uri);return f.cat_file(k)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);ap.add_argument('--review-sha256',required=True);a=ap.parse_args();os.chdir(C);assert not OUT.exists(),'Consumed journal: reconcile read-only, no blind retry';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 expected={'review-receipt.json':'6dd6946945383ae264ec888d40f34c8e74ed349cfdaba8eaa6567c5706f646e3','launch-plan.json':'8c136ca8d05facff1ea046559d712e14ba58c20631c3360f8573e3b46d7b5539'}
 for name,digest in expected.items():assert sha((D/name).read_bytes())==digest
 vr=(D/'science-verdict-marten-r1.json').read_bytes();assert sha(vr)==a.review_sha256;review=json.loads(vr);assert review['verdict']==REQUIRED_VERDICT and review['source_pin']==PIN and review['receipt_sha256']==expected['review-receipt.json'] and review['plan_sha256']==expected['launch-plan.json'] and review['native_root_bundle_sha256']==BUNDLE
 for path,item in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  raw=Path(path).read_bytes();assert len(raw)==item['bytes'] and sha(raw)==item['sha256']
 plan=json.loads((D/'launch-plan.json').read_bytes());assert len(plan['arms'])==1 and plan['training']['updates']==36 and plan['training']['learning_rate']==2e-6;target=plan['arms'][0];assert target['arm']=='A10' and target['checkpoint_updates']==[36] and target['selected_ON_occurrences']==44 and target['padding_only_sequences']==0 and target['all36_batches_nonempty_supervision'] and not target['restore_from_checkpoint'];assert target['final_checkpoint']['native_checkpoint'].endswith('/step-35')
 for key in ['current_input_manifest','current_staging_receipt']:assert sha(Path(plan[key]['path']).read_bytes())==plan[key]['sha256']
 assert sha(Path(plan['runtime']['local']).read_bytes())==plan['runtime']['sha256']=='565cf0b2e09b0e1d58b647b698364a04d41b80615b47f8e6e1dbc7167f0d8e9e';stage=json.loads(Path(plan['current_staging_receipt']['path']).read_bytes());assert stage['state']=='ALL3_FULLGET_BYTE_SHA_LENGTH_PASS_DATA_ONLY' and stage['data_bytes']==3326925
 orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A10 TULU-STYLE' in orders;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ['CW_KEY_ID','CW_KEY_SECRET']}
 for item in stage['files']:
  raw=get(item['uri']);assert len(raw)==item['bytes'] and sha(raw)==item['sha256']
 for key in ['source_receipt','source_objects']:assert sha(get(plan[key]['path']))==plan[key]['sha256']
 spec=importlib.util.spec_from_file_location('sealed_A10',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kwargs=mod.request(credentials);fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert not args and kw['training']['arm']=='A10' and kw['training']['training_pin']==PIN and kw['training']['runtime_bytes']==Path(plan['runtime']['local']).read_bytes() and kwargs['entrypoint'].workdir_files;assert kwargs['name']==target['root_name'] and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['timeout'].to_seconds()==176400
 bundle=create_workspace_zip(C);assert sha(bundle)==BUNDLE;run_root=target['training_kwargs']['run_root'];f,k=filesystem_for(run_root);assert not f.exists(k) and not f.find(k);expected_job='/karan/'+target['root_name']
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as client:
   assert not client.list_jobs(prefix=expected_job)
   record={'state':'PRE_PROBE','utc':utc(),'authority':'Jerboa A10 GO + exact combined source/data/native CLEAR, no A9/SWE/math/KEEP/baseline dependency','source_pin':PIN,'sealed_files':expected,'review_sha256':a.review_sha256,'orders_sha256_submission_only':sha(orders),'local_native_bundle_sha256':BUNDLE,'staging_sha256':plan['current_staging_receipt']['sha256'],'expected_job':expected_job,'run_root':run_root,'native_callable_serialization_preserved':True,'fresh_v01_no_resume':True,'completed_updates_requested':36,'only_final_native_step':35,'selected_originals':44,'identity_occurrences':440,'identity_OFF_ON':[396,44],'replay_occurrences':1648,'padding_sequences':0,'identity_present_updates':[1,36],'actual_dose':target['actual_dose'],'dose_comparison_reported_before_submit':plan['dose_comparison'],'scientific_limits':plan['comparison_limits'],'no_equal_canonical_or_replay_dose_or_pure_ON_effect_claim':True,'CPU_probe_before_single8H100_2h_export_then502audit_exact_TEMP_retirement':True,'standing_followups':'auditedHFthen5BFCL+original132+frozenoriginalslot1; native-boundary target separatelybound, no P1','application_submit_calls':0,'automatic_retry':False,'credentials_in_receipt':False}
   def save():
    with OUT.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   with OUT.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   probe=secrets.token_bytes(32);uri='s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/A10-submit-probes-cf4da6f40326/'+secrets.token_hex(8)+'.bin';f,k=filesystem_for(uri);assert not f.exists(k);record['write_probe']={'uri':uri,'bytes':32,'sha256':sha(probe),'state':'PUT_CALLED'};save();f.pipe_file(k,probe);assert f.cat_file(k)==probe;record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders and sha((D/'science-verdict-marten-r1.json').read_bytes())==a.review_sha256;record.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Read-only reconciliation only, no blind retry')from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expected_job;log=(D/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save()
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request;constraints=[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints];record['actual_native_request']={'bundle_id':q.bundle_id,'constraints':constraints,'priority':job_pb2.PriorityBand.Name(q.priority_band),'failure_retries':q.max_retries_failure,'preemption_retries':q.max_retries_preemption,'timeout_ms':q.timeout.milliseconds};record['actual_native_binding_PASS']=q.bundle_id==BUNDLE and 'cw-rno2a' in json.dumps(constraints) and q.priority_band==job_pb2.PRIORITY_BAND_INTERACTIVE and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==176400000;save()
   except Exception as error:record['native_readback_error_type']=type(error).__name__;save()
   print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),'job':record['job'],'notifier_pid':proc.pid,'actual_native_binding_PASS':record.get('actual_native_binding_PASS')}),flush=True)
if __name__=='__main__':main()
