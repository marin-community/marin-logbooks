"""One-shot submit; exclusive journal, no automatic retry, no model/GPU/delete calls."""
import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from botocore.config import Config
from botocore.session import get_session
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2, job_pb2
from rigging.filesystem.cluster_config import StoreType
from rigging.filesystem.s3_compat import s3_endpoint
from rigging.secrets import require_secret
from experiments.post_training.identity_path_b import hf_archive_extra_arms as worker

D=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b/public-HF-upload-extra-arms-r1')
W=Path('/home/karan/oa/marin-path-b-hf-archive-r4')
N=D.parent.parent/'t310/current-orders.md'
J=D/'actual-CPU-upload-launch-margay-r1.json'


def sha(raw): return hashlib.sha256(raw).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--orders-sha256',required=True);args=parser.parse_args()
    os.chdir(W)
    orders=N.read_bytes();assert sha(orders)==args.orders_sha256
    assert b'PUBLIC HF UPLOAD OF EXTRA ARMS' in orders
    proof=json.loads((D/'native-binding-proof.json').read_bytes())
    plan=json.loads((D/'launch-plan.json').read_bytes())
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['code_pin']==proof['code_pin']
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    assert sha(create_workspace_zip(W))==proof['native_bundle_sha256']
    receipt_sha=sha((D/'receipt.json').read_bytes())
    load_dotenv('/home/karan/oa/marin/.env')
    credentials={key:require_secret(key)for key in ['HF_TOKEN','CW_KEY_ID','CW_KEY_SECRET']}
    spec=importlib.util.spec_from_file_location('sealed_extra_HF_request',D/'prepare_request.py');packet=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet)
    kwargs=packet.request(credentials)
    inspect.signature(IrisClient.submit).bind(None,**kwargs)
    assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0
    assert kwargs['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE
    assert not kwargs['environment'].env_vars and set(kwargs['environment'].secrets)==set(credentials)
    expected='/karan/'+kwargs['name']
    r={'state':'PREFLIGHT','utc':utc(),'expected_job':expected,'application_submit_calls':0,
       'orders_sha256_submission_only':sha(orders),'receipt_sha256':receipt_sha,'code_pin':plan['code_pin'],
       'native_bundle_sha256':proof['native_bundle_sha256'],'resources':plan['resources'],
       'report_root':plan['report_root'],'card_clears':plan['card_clears'],'automatic_retry':False,
       'no_deletion':True,'no_GPU':True,'secrets_names_only':list(credentials)}
    # Any existing journal is consumed, including zero-call preflights: reconcile before a fresh attempt.
    with J.open('x')as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
    def save():
        with J.open('w')as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    client_s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',
        aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],
        config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=60,retries={'total_max_attempts':1},
                      request_checksum_calculation='when_required',response_checksum_validation='when_required'))
    report_key=plan['report_root'].removeprefix('s3://marin-rno2a/')
    assert not client_s3.list_objects_v2(Bucket='marin-rno2a',Prefix=report_key+'/',MaxKeys=1).get('Contents')
    config=json.loads((worker.ARTIFACTS/worker.TARGETS).read_bytes())
    for t in config['targets']:
        bound=json.loads(worker.checked_artifact(t['inventory_file'],t['inventory_sha256']))
        actual=worker.source_listing(client_s3,t['source'])
        assert actual=={n:{'bytes':v['bytes'],'etag':v['etag']}for n,v in bound.items()},t['label']
        for name,digest in t['admitted_metadata_sha256'].items():
            b,k=worker.s3_location(t['source'])
            with client_s3.get_object(Bucket=b,Key=k+'/'+name)['Body'] as stream: raw=stream.read()
            assert sha(raw)==digest,(t['label'],name)
    key='tmp/ttl=14d/snowball-identity/path-b/extra-arms-upload-probes/'+secrets.token_hex(8)+'.bin'
    raw=secrets.token_bytes(32)
    r['write_probe']={'key':key,'sha256':sha(raw),'state':'PUT_CALLED'};save()
    client_s3.put_object(Bucket='marin-rno2a',Key=key,Body=raw)
    with client_s3.get_object(Bucket='marin-rno2a',Key=key)['Body']as stream:assert stream.read()==raw
    r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save();client_s3.close()
    with connect_controller(cluster_name='marin')as co:
        with IrisClient.remote(co.url,workspace=W,credentials=co.credentials)as client:
            assert not client.list_jobs(prefix=expected),'Job root already exists; reconcile, never retry'
            assert N.read_bytes()==orders,'Orders changed before submit; zero calls'
            assert sha((D/'receipt.json').read_bytes())==receipt_sha
            r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
            try: job=client.submit(**kwargs)
            except Exception as e:
                r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(e).__name__);save()
                raise RuntimeError('Read-only reconciliation required; no automatic retry') from None
            r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save()
            assert r['job']==expected
            env=dict(os.environ)
            with (D/'completion-notifier.log').open('ab')as log:
                proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py',
                    '--job',r['job'],'--timeout','45000','--out',str(D/'completion-native.json')],env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            r['one_completion_notifier_pid']=proc.pid;save()
            try:
                q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request
                r['host_root_stored_request']={'bundle_id':q.bundle_id,'constraints':[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints],
                    'priority':job_pb2.PriorityBand.Name(q.priority_band),'timeout_ms':q.timeout.milliseconds,
                    'resources':MessageToDict(q.resources,preserving_proto_field_name=True),
                    'max_retries_failure':q.max_retries_failure,'max_retries_preemption':q.max_retries_preemption}
                r['host_root_bundle_verified']=q.bundle_id==proof['native_bundle_sha256'];save()
            except Exception as e:r['post_submit_readback_error_type']=type(e).__name__;save()
    print(json.dumps({'job':r['job'],'one_notifier_pid':r['one_completion_notifier_pid'],'receipt':str(J),'host_root_bundle_verified':r.get('host_root_bundle_verified')}),flush=True)


if __name__=='__main__':main()
