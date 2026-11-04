"""Authorized1CPU/120s import-only proof, exact native committed workspace bundle."""
import hashlib,inspect,io,json,os,subprocess,sys,datetime,zipfile
from pathlib import Path
from dotenv import load_dotenv
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2
N=Path('/home/karan/Dropbox/marin-notes/snowball-identity');D=N/'path-b/full-screen-bundled-r4';C=Path('/home/karan/oa/marin-path-b-screen-owner');PIN='dce28ebcd38e63f7c175b8769f8133fe6ffc08a9';OUT=D/'import-probe-launch-margay-r1.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();assert not OUT.exists(),'Consumed importprobejournal'
 raw=(D/'receipt.json').read_bytes();assert sha(raw)=='ebe3b2d1e64d4c24d729f521adfbf1667d484c5bf66e53257497ee95c84d579e';receipt=json.loads(raw)
 for f in receipt['files']:
  b=Path(f['path']).read_bytes();assert len(b)==f['bytes'] and sha(b)==f['sha256'],f['path']
 binding=json.loads((D/'binding.json').read_bytes());assert sha((D/'binding.json').read_bytes())==receipt['binding_sha256']=='03e35e26bda463d0706d7028cd14593dc665c8b7597d77b6a5884ca2e8d7cef7'
 bundle=create_workspace_zip(C);assert len(bundle)==binding['native_bundle_bytes'] and sha(bundle)==binding['native_bundle_sha256']=='71a1820e51ab1d40b9f51a1461a86a678168faacd5e6ad0c0be0d497a0734ea9'
 with zipfile.ZipFile(io.BytesIO(bundle)) as z:
  for f in binding['package_members']:assert len(z.read(f['path']))==f['bytes'] and sha(z.read(f['path']))==f['sha256']
 sys.path.insert(0,str(D));from prepare_requests import probe_request
 spec=probe_request(binding['probe_name']);inspect.signature(IrisClient.submit).bind(object(),**spec);assert spec['timeout'].to_seconds()==120 and spec['resources'].to_proto().cpu_millicores==1000 and spec['max_retries_failure']==spec['max_retries_preemption']==0 and not spec['environment'].secrets
 orders_path=N/'t310/current-orders.md';orders=orders_path.read_bytes();assert sha(orders)=='3cbf5c11c48f099c9e2d9b7e4b16f3b17deac5d56c0b7638b11e04197b36311d';load_dotenv('/home/karan/oa/marin/.env')
 record={'state':'PRE_SUBMIT','utc':utc(),'authority':'Jerboa explicitly1CPU2min samebundle import-only proof before singlefresh52r3; no model/children/storage actions by application','code_pin':PIN,'binding_sha256':receipt['binding_sha256'],'packet_receipt_sha256':sha(raw),'local_native_bundle_sha256':sha(bundle),'expected_job':'/karan/'+spec['name'],'orders_sha256_submission_only':sha(orders),'requested_CPU':1,'requested_timeout_seconds':120,'application_submit_calls':0,'no_automatic_retry':True,'owner_r3_not_yet_authorized_by_this_script':True}
 def save():
  with OUT.open('w') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 with connect_controller(cluster_name='marin') as conn:
  with IrisClient.remote(conn.url,workspace=C,credentials=conn.credentials) as client:
   assert not client.list_jobs(prefix=record['expected_job'])
   with OUT.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   assert orders_path.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',application_submit_calls=1,submit_called_utc=utc());save()
   try:job=client.submit(**spec)
   except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile exactnamespace read-only; no automaticretry') from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==record['expected_job'];save()
   request=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request;record['actual_bundle_id']=request.bundle_id;assert request.bundle_id;save()
 print(json.dumps({'receipt':str(OUT),'receipt_sha256':sha(OUT.read_bytes()),**record}),flush=True)
if __name__=='__main__':main()
