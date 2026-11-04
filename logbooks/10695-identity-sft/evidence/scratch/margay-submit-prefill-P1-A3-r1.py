"""ONE reviewed P1 A3 native think-prefill identity132 screen; existing HF only."""
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'prefill-p1-r1';C=Path('/home/karan/oa/marin-path-b-prefill-p1');N=P.parent/'t310/current-orders.md'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);a=ap.parse_args();os.chdir(C)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='1e1b82d032b52926430e3d4fa377fffe0226e148';assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 assert sha(a.review.read_bytes())==a.review_sha256;review=json.loads(a.review.read_bytes());assert review['verdict']=='CLEAR_CAPTURE_EVAL_ONLY' and review['code_pin']=='1e1b82d032b52926430e3d4fa377fffe0226e148' and review['plan_sha256']=='c4f8254c5826f710891d6001c43481f78cdaf961dcbfbed79951808bbcbdf791' and review['receipt_sha256']=='e8ea3e495e0ee98f0677f5c5e76fc66b4af63ff98beafc1e9ab3817cb8292257' and review['native_bundle_sha256']=='217164baf686b60249870e6e85f4241ea950957407836f49e983a89739edb705'
 assert sha((D/'receipt.json').read_bytes())=='e8ea3e495e0ee98f0677f5c5e76fc66b4af63ff98beafc1e9ab3817cb8292257';assert sha((D/'launch-plan.json').read_bytes())=='c4f8254c5826f710891d6001c43481f78cdaf961dcbfbed79951808bbcbdf791'
 plan=json.loads((D/'launch-plan.json').read_bytes());proof=json.loads((D/'native-binding-proof.json').read_bytes());assert proof['native_bundle_sha256']=='217164baf686b60249870e6e85f4241ea950957407836f49e983a89739edb705' and proof['no_workdir_files']
 for item in plan['metadata_proofs']:assert sha(Path(item['path']).read_bytes())==item['sha256']
 assert plan['prefill']['native_ON_prefix_addition']==[128002] and plan['prefill']['ON_prefix_count']==plan['prefill']['OFF_prefix_identical']==66 and plan['prefill']['server_tokenize_before_each_ON_POST'] and not plan['prefill']['parser_repair']
 spec=importlib.util.spec_from_file_location('sealed_P1_A3_identity132',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ('CW_KEY_ID','CW_KEY_SECRET')};target=plan['targets']['A3'];original=json.loads((P/'bfcl-ablation-r1/A3/bfcl-launch.json').read_bytes())
 for name,expected in [('path-b-checkpoint-event.json',original['event_sha256']),('path-b-storage-audit.json',original['audit_sha256']),('chat_template.jinja','b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de'),('tokenizer_config.json','d9270d9407a15ec895fa2cdb0b6fbe9cd7dab937b91c2db708a918b35efa18b3')]:
  f,k=filesystem_for(target['event']['hf_export']+'/'+name);assert sha(f.cat_file(k))==expected
 f,k=filesystem_for(target['output']);assert not f.exists(k) and not f.find(k);out=D/'A3-launch.json';assert not out.exists(),'Consumedjournal;readonlyreconcile'
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as client:
   orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'P1, eval only' in orders;kwargs=mod.request('A3',credentials);expected='/karan/'+target['root_name'];assert kwargs['name']==target['root_name'] and kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE;assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;assert not client.list_jobs(prefix=expected)
   record={'state':'PRE_SUBMIT','utc':utc(),'arm':'A3','intervention':'P1_NATIVE_ON_START_THINK_PREFILL_ONLY','authority':'JerboaP1conditionalGO+MartenbindingCLEAR;ONEevalonlyidentity132;NOtraining/reexport','code_pin':plan['code_pin'],'plan_sha256':sha((D/'launch-plan.json').read_bytes()),'packet_sha256':sha((D/'receipt.json').read_bytes()),'review':str(a.review),'review_sha256':a.review_sha256,'expected_native_bundle_sha256':proof['native_bundle_sha256'],'orders_sha256_submission_only':sha(orders),'expected_job':expected,'output':target['output'],'exact_target_event':target['event'],'protocol':plan['protocol'],'prefill':plan['prefill'],'application_submit_calls':0,'no_native_checkpoint_ops':True,'no_training_or_reexport':True,'no_parser_or_output_repair':True,'runtime_tokenize_PASS_not_yet_observed':True,'no_automatic_retry':True,'credentials_in_receipt':False}
   def save():
    with out.open('w') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   assert N.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
   try:job=client.submit(**kwargs)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Readonlyreconcile;neverautomaticallyretry') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();log=(D/'A3-completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'A3-completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'intervention':'P1_A3_NATIVE_THINK_PREFILL','notifier_pid':proc.pid}),flush=True)
if __name__=='__main__':main()
