"""Single authorized dataset archive submit; exclusive journal and no automatic retry."""
import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2, job_pb2
from rigging.secrets import require_secret
from experiments.post_training.identity_path_b import hf_inputs_archive as worker

D=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b/A13-inputs-HF-r1')
N=D.parent.parent/'t310/current-orders.md'
J=D/'actual-CPU-dataset-upload-launch-margay-r1.json'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
 p=argparse.ArgumentParser();p.add_argument('--orders-sha256',required=True);args=p.parse_args()
 orders=N.read_bytes();assert sha(orders)==args.orders_sha256 and b'A13 INPUTS TO HF' in orders
 plan=json.loads((D/'launch-plan.json').read_bytes());proof=json.loads((D/'native-binding-proof.json').read_bytes());W=Path(plan['workspace']);os.chdir(W)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['code_pin']==proof['code_pin']
 assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 assert sha(create_workspace_zip(W))==proof['native_bundle_sha256']
 for item in plan['S3_preservation_receipts']:
  raw=Path(item['path']).read_bytes();assert sha(raw)==item['sha256']
  rec=json.loads(raw);assert rec['state']in ('FOUR_INPUTS_AND_DOCUMENTS_FULL_BYTE_VERIFIED','ALL_RECIPE_DOCUMENTS_FULL_BYTE_VERIFIED')
 config=json.loads((worker.ARTIFACTS/'upload-config.json').read_bytes());assert config['files']==plan['files']
 for item in config['files']:
  if 'artifact'in item:assert sha((worker.ARTIFACTS/item['artifact']).read_bytes())==item['sha256']
 receipt_sha=sha((D/'receipt.json').read_bytes())
 load_dotenv('/home/karan/oa/marin/.env');creds={k:require_secret(k)for k in ('HF_TOKEN','CW_KEY_ID','CW_KEY_SECRET')}
 spec=importlib.util.spec_from_file_location('dataset_packet',D/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);req=m.request(creds)
 inspect.signature(IrisClient.submit).bind(None,**req)
 assert req['max_retries_failure']==req['max_retries_preemption']==0 and not req['environment'].env_vars
 assert req['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE
 expected='/karan/'+req['name'];r=dict(state='BEFORE_SUBMIT',utc=utc(),expected_job=expected,application_submit_calls=0,
  orders_sha256_submission_only=sha(orders),code_pin=plan['code_pin'],native_bundle_sha256=proof['native_bundle_sha256'],
  requested_resources=plan['requested_resources'],repo_id=plan['repo_id'],repo_type='dataset',
  attribution_sha256=plan['attribution_sha256'],admission_sha256=plan['admission_sha256'],
  report_root=plan['report_root'],no_deletion=True,automatic_retry=False)
 with J.open('x')as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
 def save():
  with J.open('w')as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=W,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expected),'Existing root: reconcile, do not retry'
   assert N.read_bytes()==orders,'Orders changed before submit: zero application submit calls'
   assert sha((D/'receipt.json').read_bytes())==receipt_sha
   r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**req)
   except Exception as e:
    r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save();raise RuntimeError('Read-only reconciliation required')from None
   r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job']==expected
   with (D/'completion-notifier.log').open('ab')as log:
    proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','4500','--out',str(D/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   r['one_completion_notifier_pid']=proc.pid;save()
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request
    r['host_root_stored_request']=dict(bundle_id=q.bundle_id,constraints=[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints],resources=MessageToDict(q.resources,preserving_proto_field_name=True),timeout_ms=q.timeout.milliseconds,priority=job_pb2.PriorityBand.Name(q.priority_band),max_retries_failure=q.max_retries_failure,max_retries_preemption=q.max_retries_preemption)
    r['host_root_bundle_verified']=q.bundle_id==proof['native_bundle_sha256'];save()
   except Exception as e:r['post_submit_readback_error_type']=type(e).__name__;save()
 print(json.dumps(dict(job=r['job'],one_notifier_pid=r['one_completion_notifier_pid'],host_root_bundle_verified=r.get('host_root_bundle_verified'))),flush=True)


if __name__=='__main__':main()
