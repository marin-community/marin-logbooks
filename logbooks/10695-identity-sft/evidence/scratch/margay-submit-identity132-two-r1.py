"""Submit only the two sealed authorized identity132 roots, once each."""
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'ablation-identity132-r1';C=Path('/home/karan/oa/marin-path-b-ablation-identity');N=P.parent/'t310/current-orders.md'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);args=ap.parse_args();os.chdir(C)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='ac986b08026c07ab5096535361bccc36bf01d553'
 assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 assert sha((D/'receipt.json').read_bytes())=='3b8d143317009366dccf4200207cc68ea03e5b34709df47a6efe46e9be723c9d'
 assert sha((D/'launch-plan.json').read_bytes())=='38e25c2f23e35f5aaf1e4d4329503cd0a0254bb1ec8ba647ef02bda6952f2bce'
 plan=json.loads((D/'launch-plan.json').read_bytes());spec=importlib.util.spec_from_file_location('sealed_identity132',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k) for k in ('CW_KEY_ID','CW_KEY_SECRET')}
 with connect_controller(cluster_name='marin') as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials) as client:
   for arm in ('A0','A3'):
    orders=N.read_bytes();assert sha(orders)==args.orders_sha256;target=plan['targets'][arm];out=D/f'{arm}-launch.json';assert not out.exists();kwargs=mod.request(arm,credentials);expected='/karan/'+target['root_name'];assert kwargs['name']==target['root_name'];assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;assert not client.list_jobs(prefix=expected)
    record={'state':'PRE_SUBMIT','utc':utc(),'arm':arm,'code_pin':plan['code_pin'],'plan_sha256':sha((D/'launch-plan.json').read_bytes()),'packet_sha256':sha((D/'receipt.json').read_bytes()),'orders_sha256_submission_only':sha(orders),'expected_job':expected,'output':target['output'],'exact_target_event':target['event'],'protocol':plan['protocol'],'application_submit_calls':0,'no_native_checkpoint_ops':True,'no_extra_science_review_gate':True,'no_automatic_retry':True,'credentials_in_receipt':False}
    def save():
     with out.open('w') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    assert N.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',called_utc=utc(),application_submit_calls=1);save()
    try:job=client.submit(**kwargs)
    except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile readonly; never automatically retry') from None
    record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();log=(D/f'{arm}-completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/f'{arm}-completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'arm':arm,'notifier_pid':proc.pid}),flush=True)
if __name__=='__main__':main()
