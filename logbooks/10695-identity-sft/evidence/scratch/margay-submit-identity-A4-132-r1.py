"""One authorized A4 identity132 root on the existing audited HF; no re-export."""
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-identity-a4-r1';C=Path('/home/karan/oa/marin-path-b-identity-a4');N=P.parent/'t310/current-orders.md'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='ef8234140f041987d9d47cca668b8fba4fa4a0ec';assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 assert sha((D/'receipt.json').read_bytes())=='2b115be76c0b68b4df6fecbf27671172e1349ccd9798bbb8738dee4254762056';assert sha((D/'launch-plan.json').read_bytes())=='3263cef00cb191201308b57aa6a706714f20630d6c58394248c2eb3f873f18bf'
 plan=json.loads((D/'launch-plan.json').read_bytes());proof=json.loads((D/'native-binding-proof.json').read_bytes());assert proof['native_bundle_sha256']=='a27ca2aea63164af30e05db71d91c2bca3d537786de84c6e1a52c1560766533e' and proof['no_workdir_files'];assert sha(Path(plan['model_existence']['path']).read_bytes())==plan['model_existence']['sha256']
 spec=importlib.util.spec_from_file_location('sealed_A4_identity132',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ('CW_KEY_ID','CW_KEY_SECRET')};target=plan['targets']['A4'];original=json.loads((P/'bfcl-ablation-r1/A4/bfcl-launch.json').read_bytes())
 for name,key in [('path-b-checkpoint-event.json','event_sha256'),('path-b-storage-audit.json','audit_sha256')]:
  f,k=filesystem_for(target['event']['hf_export']+'/'+name);assert sha(f.cat_file(k))==original[key]
 f,k=filesystem_for(target['output']);assert not f.exists(k) and not f.find(k);out=D/'A4-launch.json';assert not out.exists(),'Consumedjournal;readonlyreconcile'
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as client:
   orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'A4 identity screen' in orders;kwargs=mod.request('A4',credentials);expected='/karan/'+target['root_name'];assert kwargs['name']==target['root_name'] and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE;assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;assert not client.list_jobs(prefix=expected)
   record={'state':'PRE_SUBMIT','utc':utc(),'arm':'A4','authority':'JerboaNEXTONEA4same132identityscreen;existingHFonly;ONEattempt','code_pin':plan['code_pin'],'plan_sha256':sha((D/'launch-plan.json').read_bytes()),'packet_sha256':sha((D/'receipt.json').read_bytes()),'expected_native_bundle_sha256':proof['native_bundle_sha256'],'orders_sha256_submission_only':sha(orders),'expected_job':expected,'output':target['output'],'exact_target_event':target['event'],'protocol':plan['protocol'],'application_submit_calls':0,'no_native_checkpoint_ops':True,'no_reexport':True,'no_extra_science_review_gate':True,'no_automatic_retry':True,'credentials_in_receipt':False}
   def save():
    with out.open('w') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   assert N.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile readonly;neverautomaticallyretry') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();log=(D/'A4-completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'A4-completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'arm':'A4','notifier_pid':proc.pid}),flush=True)
if __name__=='__main__':main()
