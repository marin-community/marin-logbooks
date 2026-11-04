"""ONE reviewed A7 fixed200 A3-NOSYS-native-prefill pilot; no full capture or training."""
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
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-a7-pilot-r1';C=Path('/home/karan/oa/marin-path-b-a7-pilot');N=P.parent/'t310/current-orders.md';J=D/'launch-margay-r1.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);a=ap.parse_args();os.chdir(C);assert not J.exists(),'Consumedpilotjournal;NOretry'
 vr=a.review.read_bytes();assert sha(vr)==a.review_sha256;v=json.loads(vr);assert v['verdict']=='CLEAR_ONE_A7_200_SOURCE_PILOT_ONLY';assert v['code_pin']=='6be498f450c083fc784cc1222e775558bf2a23d3' and v['plan_sha256']=='f65b3d737f974d8c41b37a754afca647a93b21f4d9ed48db39f0b85691bb9d2b' and v['receipt_sha256']=='1eb979a37914c388d205660f645da387b2195964daa9c332d03c28d9cc96f4db'
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==v['code_pin'] and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert sha((D/'launch-plan.json').read_bytes())==v['plan_sha256'] and sha((D/'receipt.json').read_bytes())==v['receipt_sha256'];plan=json.loads((D/'launch-plan.json').read_bytes());bound=plan['binding'];assert len(bound['source_ids'])==len(set(bound['source_ids']))==200 and len(bound['family_quotas'])==11 and sum(bound['family_quotas'].values())==200;assert bound['minimum_kept_overall']==50 and bound['minimum_kept_every_family']==2 and bound['teacher_system']==''
 assert plan['staging_receipt']['sha256']=='09810bf17eba32e2f1664f16cca3f2b0a64302a098e8ae629000dc05c8997786' and sha(Path(plan['staging_receipt']['path']).read_bytes())==plan['staging_receipt']['sha256'];bundle=create_workspace_zip(C);bundle_sha=sha(bundle);assert bundle_sha==v['native_bundle_sha256']=='1c543d24a7f8534bf7212f7ff13097d15bc1643addcfe2e02a2fdf1d480e71ea'
 orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A7 PILOT' in orders;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ('CW_KEY_ID','CW_KEY_SECRET')};spec=importlib.util.spec_from_file_location('reviewed_A7_request',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kwargs=mod.request(credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);assert kwargs['name']==bound['run_id'] and kwargs['timeout'].to_seconds()==43200 and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0 and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and kwargs['replicas']==1;fs,key=filesystem_for(bound['artifact_root']);assert not fs.exists(key) and not fs.find(key)
 for name,expected in bound['teacher_metadata_sha256'].items():
  f,k=filesystem_for(bound['teacher_event']['hf_export']+'/'+name);assert sha(f.cat_file(k))==expected
 now=datetime.datetime.now(datetime.timezone.utc);probe_uri='s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/a7-teacher-pilot-20261030-r1/probes/pre-submit-'+now.strftime('%Y%m%dT%H%M%S.%fZ')+'.bin';pb=os.urandom(32);f,k=filesystem_for(probe_uri);assert not f.exists(k);f.pipe_file(k,pb);assert f.cat_file(k)==pb;m=bound['manifest'];raw=Path(m['local']).read_bytes();assert len(raw)==m['bytes']==122407 and sha(raw)==m['sha256']=='3da2d4934c9cacfc016704508cc7b3b06a0a79b0b61d60d7a77127167f79e9fb';f,k=filesystem_for(m['uri']);assert f.cat_file(k)==raw;stage={'utc':utc(),'probe':{'uri':probe_uri,'bytes':32,'sha256':sha(pb),'FULLGET_equal':True},'manifest':m,'existing_staged_manifest_FULLGET_byte_equal':True,'no_manifest_rewrite':True,'no_model_GPU_calls':True,'not_quota_or_future_write_guarantee':True};sp=D/'pre-submit-CPU-probe-and-input-FULLGET-margay-r1.json';b=(json.dumps(stage,indent=2)+'\n').encode()
 with sp.open('xb') as h:h.write(b)
 record={'state':'PRE_SUBMIT','opened_utc':utc(),'code_pin':v['code_pin'],'CLEAR_sha256':sha(vr),'plan_sha256':v['plan_sha256'],'packet_sha256':v['receipt_sha256'],'CPU_probe_stage_receipt':str(sp),'CPU_probe_stage_sha256':sha(b),'original_CPU_staging_receipt_sha256':plan['staging_receipt']['sha256'],'orders_sha256_submission_only':sha(orders),'local_native_bundle_sha256':bundle_sha,'native_bundle_bytes':len(bundle),'expected_job':'/karan/'+bound['run_id'],'artifact_root':bound['artifact_root'],'fixed_pilot_manifest':m,'originals':200,'family_quotas':bound['family_quotas'],'minimum_kept_overall':50,'minimum_kept_every_family':2,'scope':'ONEA7PILOTONLY;NO fullcapture/pool/training/A5revival','teacher_HF':bound['teacher_event']['hf_export'],'teacher_NOSYS_native_prefill':True,'teacher_metadata_sha256':bound['teacher_metadata_sha256'],'named_prefill_student_graft_NOT_raw_only_T2_claim':True,'no_output_strip_trim_or_repair':True,'priority':'interactive','application_submit_calls':0,'no_retry_resample_repair_bestof_or_second_pilot':True,'credentials_in_receipt':False}
 def save():
  with J.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as client:
   assert not client.list_jobs(prefix=record['expected_job'])
   with J.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
   assert N.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Readonlyreconcile;NOsecondpilot') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==record['expected_job'];save();log=(D/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','46800','--out',str(D/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save()
   try:
    req=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request;record.update(actual_bundle_id=req.bundle_id,actual_bundle_matches_local=req.bundle_id==bundle_sha);save()
   except Exception as error:record['bundle_readback_error_type']=type(error).__name__;save()
 print(json.dumps({'receipt':str(J),'sha256':sha(J.read_bytes()),'state':record['state'],'job':record['job'],'actual_bundle_matches_local':record.get('actual_bundle_matches_local'),'application_submit_calls':1}),flush=True)
if __name__=='__main__':
 raise SystemExit('BLOCKED: A7 draft6be/r1 has NO pilot CLEAR (name-negation and returned-prompt-ID gaps). NEVER submit. Use the separately sealed r2 successor only after NEW Marten pilot CLEAR; still ONE actual pilot total.')
