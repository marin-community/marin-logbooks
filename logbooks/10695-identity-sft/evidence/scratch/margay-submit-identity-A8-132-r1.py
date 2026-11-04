"""One ordered ORIGINAL identity132 per A8 HF after its actual502 audit; NO P1."""
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-identity-a8-r1';C=Path('/home/karan/oa/marin-path-b-identity-a8');N=P.parent/'t310/current-orders.md'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['A8a','A8b'],required=True);ap.add_argument('--readiness-sha256',required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='6f3d23355c7a5a44b0c50817fb537ffb1e0a7d2f';assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 assert sha((D/'receipt.json').read_bytes())=='da39bd612df24c4019cc277e2bdff70945c98367a3cc6569baf556a5d820f674';assert sha((D/'launch-plan.json').read_bytes())=='7b0f19af5a79c772d712f6a8ef9961331bfdc81055547053f94656ce82e75f12'
 plan=json.loads((D/'launch-plan.json').read_bytes());proof=json.loads((D/'native-binding-proof.json').read_bytes());assert proof['native_bundle_sha256']=='ba62f61023b8086cb9001f3a871d617156e9fe439c627fc55b6b086c7cddb91b' and next(t for t in proof['arms'] if t['arm']==a.arm)['no_small_workdir_files'];up=P/'ablation-a8-r1'/a.arm/'actual-audited-HF-ready-margay-r1.json';assert sha(up.read_bytes())==a.readiness_sha256;admitted=json.loads(up.read_bytes());assert admitted['actual_bfcl_ready_marker']
 spec=importlib.util.spec_from_file_location('sealed_A8_identity132',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ('CW_KEY_ID','CW_KEY_SECRET')};target=plan['targets'][a.arm];original={'event_sha256':admitted['metadata']['event.json']['sha256'],'audit_sha256':admitted['metadata']['audit.json']['sha256']}
 for name,key in [('path-b-checkpoint-event.json','event_sha256'),('path-b-storage-audit.json','audit_sha256')]:
  f,k=filesystem_for(target['event']['hf_export']+'/'+name);assert sha(f.cat_file(k))==original[key]
 f,k=filesystem_for(target['output']);assert not f.exists(k) and not f.find(k);out=D/f'{a.arm}-launch.json';assert not out.exists(),'Consumedjournal;readonlyreconcile'
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as client:
   orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'ON-share dose arms A8a / A8b' in orders;kwargs=mod.request(a.arm,credentials);expected='/karan/'+target['root_name'];assert kwargs['name']==target['root_name'] and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE;assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;assert not client.list_jobs(prefix=expected)
   record={'state':'PRE_SUBMIT','utc':utc(),'arm':a.arm,'authority':'JerboaA8GO+Marten6fc95936ORIGINALidentity132AFTERactual502audit;ONEattempt;NO P1prefill','code_pin':plan['code_pin'],'plan_sha256':sha((D/'launch-plan.json').read_bytes()),'packet_sha256':sha((D/'receipt.json').read_bytes()),'expected_native_bundle_sha256':proof['native_bundle_sha256'],'orders_sha256_submission_only':sha(orders),'expected_job':expected,'output':target['output'],'exact_target_event':target['event'],'protocol':plan['protocol'],'application_submit_calls':0,'no_native_checkpoint_ops':True,'no_reexport':True,'no_extra_science_review_gate':True,'no_automatic_retry':True,'credentials_in_receipt':False}
   def save():
    with out.open('w') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   assert N.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile readonly;neverautomaticallyretry') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();log=(D/f'{a.arm}-completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/f'{a.arm}-completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'arm':a.arm,'notifier_pid':proc.pid}),flush=True)
if __name__=='__main__':main()
