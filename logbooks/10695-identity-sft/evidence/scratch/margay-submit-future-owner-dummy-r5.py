"""ONE authorized CPU1/120s DUMMY candidate/persistence smoke; no scientific job."""
import datetime,hashlib,inspect,json,os,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2
from rigging.secrets import SecretBehavior
N=Path('/home/karan/Dropbox/marin-notes/snowball-identity');D=N/'path-b/full-screen-owner-binding-r5';C=Path('/home/karan/oa/marin-path-b-screen-owner');PIN='c1cea8dbf2016534b3b7984c42e1c44618bfbf5d';OUT=D/'dummy-smoke-launch-margay-r1.json';NAME='pathb-future-owner-dummy-r1-c1cea8dbf201'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert not OUT.exists(),'Consumed one-shot journal; no retry'
 raw=(D/'receipt.json').read_bytes();assert sha(raw)=='784ada0375d6408f3cb79e34a95740bc241d5aba61bdf2d9b5afeb007f7e9eba';receipt=json.loads(raw)
 for path,binding in receipt['files'].items():
  b=Path(path).read_bytes();assert len(b)==binding['bytes'] and sha(b)==binding['sha256'],path
 bundle=create_workspace_zip(C);assert len(bundle)==16701911 and sha(bundle)==receipt['native_bundle_sha256']=='da5e78e53b41931776f7a193c2683c0872d8b50c33e0de79b2947ed65f0366ef'
 proof=json.loads((D/'bundle-and-command-proof.json').read_bytes());sys.path.insert(0,str(D));from prepare_request import dummy_request
 spec=dummy_request(NAME);inspect.signature(IrisClient.submit).bind(object(),**spec);assert spec['timeout'].to_seconds()==120 and spec['resources'].to_proto().cpu_millicores==1000 and spec['max_retries_failure']==spec['max_retries_preemption']==0 and spec['environment'].secret_behavior==SecretBehavior.ISOLATE and not spec['environment'].secrets;assert spec['entrypoint'].command==proof['actual_dummy_command'] and len(spec['entrypoint'].workdir_files)==1
 for path,b in spec['entrypoint'].workdir_files.items():assert sha(b)==proof['transported_workdir_files'][path]['sha256'] and len(b)==1068 and json.loads(b)['model_path']=='DUMMY-NOT-A-MODEL'
 orders_path=N/'t310/current-orders.md';orders=orders_path.read_bytes();assert sha(orders)=='ba2c75c517210482357102b007b65c3774c4f17e961db40b800b04b2bba1bb47';load_dotenv('/home/karan/oa/marin/.env')
 record={'state':'PRE_SUBMIT','utc':utc(),'authority':'Jerboa ONE CPU1/120s future-owner dummy smoke, no Marten read;52 STOPPED/no reruns','code_pin':PIN,'packet_receipt_sha256':sha(raw),'local_native_bundle_sha256':sha(bundle),'expected_job':'/karan/'+NAME,'orders_sha256_submission_only':sha(orders),'requested_CPU':1,'requested_timeout_seconds':120,'application_submit_calls':0,'no_automatic_retry':True,'secret_behavior':'ISOLATE','actual_candidate_path':next(iter(spec['entrypoint'].workdir_files)),'candidate_sha256':proof['exact_extracted_bundle_command']['candidate_sha256'],'scope':'actual production child command in inspection-only + MemoryFileSystem persistence; DUMMY, no model/child/application-object-store calls','not_a_52_rerun':True,'scientific_jobs_submitted':0}
 def save():
  with OUT.open('w') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 with connect_controller(cluster_name='marin') as conn:
  with IrisClient.remote(conn.url,workspace=C,credentials=conn.credentials) as client:
   assert not client.list_jobs(prefix=record['expected_job'])
   with OUT.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   assert orders_path.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',application_submit_calls=1,submit_called_utc=utc());save()
   try:job=client.submit(**spec)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile exact namespace read-only; no automatic retry') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==record['expected_job'];save();request=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request;record['actual_bundle_id']=request.bundle_id;save();assert request.bundle_id==sha(bundle)
 print(json.dumps({'receipt':str(OUT),'receipt_sha256':sha(OUT.read_bytes()),**record}),flush=True)
if __name__=='__main__':main()
