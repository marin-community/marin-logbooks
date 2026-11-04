"""ONE production export-only recovery of original A12 native17; no training."""
import argparse, datetime, hashlib, importlib.util, inspect, json, os, secrets, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv
from botocore.session import get_session
from botocore.config import Config
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2, job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_endpoint
from rigging.filesystem.cluster_config import StoreType
from rigging.secrets import require_secret
P = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
D = P/'A12-export-only-recovery-r1'
C = Path('/home/karan/oa/marin-path-b-a12-export-recovery')
N = P.parent/'t310/current-orders.md'
PIN = '8125c4faa1e7bb4ed9fa35e5bb9aaf45a3602362'
TRAIN = '69ddc88fb2ff4a6e45ee5cbbec866d9f86c2d068'
ZIP = 'ada5e4fcb7635cf84d248e79d3cf42a6bf819463a1344c68abcecebae6855698'
CLEAR = 'cb426ac79e446c940e3fe9f7379c0ddfc0c8ab88761711cec71fa1a770696892'
RECEIPT = 'da120443e5aec1063f936aa3de674a8d712fab3e0e5c5f52852c5ff060d4288d'
PLAN = '28d00d849201cfd9adb91b0c6bc1784e566bf4c5f09c1f007ff7969686324f4e'
def sha(b): return hashlib.sha256(b).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
    f, k = filesystem_for(uri)
    return f.cat_file(k)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--orders-sha256', required=True); args = ap.parse_args()
    os.chdir(C); out = D/'actual-export-only-recovery-launch-margay-r1.json'
    assert not out.exists(), 'Consumed journal: reconcile read-only, never repeat'
    assert subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git','status','--porcelain'], text=True).strip()
    sealed = {'binding-review-marten-r1.json':CLEAR, 'review-receipt.json':RECEIPT, 'launch-plan.json':PLAN}
    for name, h in sealed.items(): assert sha((D/name).read_bytes()) == h
    review = json.loads((D/'binding-review-marten-r1.json').read_bytes())
    assert review['verdict'] == 'CLEAR_ONE_PRODUCTION_A12_EXPORT_ONLY_RECOVERY'
    assert (review['source_pin'],review['scientific_training_pin'],review['receipt_sha256'],review['plan_sha256'],review['native_bundle_sha256']) == (PIN,TRAIN,RECEIPT,PLAN,ZIP)
    ref = review['independent_review']; assert sha(Path(ref['path']).read_bytes()) == ref['sha256'] == '36e07463d3e136de593ed7c5d4f6203d31abee228b081bee206e7ee2727ab19d'
    for path, item in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
        raw = Path(path).read_bytes(); assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    plan = json.loads((D/'launch-plan.json').read_bytes()); item = plan['A12']; event = item['event']; root = item['root']
    assert item['mode'] == 'EXPORT_ONLY_FROM_ORIGINAL_NATIVE17' and plan['training_children'] == 0 and plan['production_owner_and_export'] is True
    assert root == 's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-a12-export-recovery-r1-20261031/A12/run'
    assert event['native_checkpoint'] == item['original_training']['run_root']+'/checkpoints/step-17'
    assert event['hf_export'] == root+'/hf/updates-18' and event['training_pin'] == TRAIN and event['completed_updates'] == 18
    assert {k for k in event if event[k] != item['original_event'][k]} == {'hf_export','eval_scratch'}
    assert {k for k in item['training'] if item['training'][k] != item['original_training'][k]} == {'run_id','run_root'}
    assert sha(Path(plan['source_proof']).read_bytes()) == plan['source_proof_sha256'] == '93c5689fc40192704ad2fb1897e8a92f3142742a8922b3e6cce28f1c425b4463'
    runtime = Path(plan['runtime']['local']).read_bytes(); assert sha(runtime) == plan['runtime']['sha256']
    orders = N.read_bytes(); assert sha(orders) == args.orders_sha256 and b'A12 = A8a at 18 updates' in orders
    load_dotenv('/home/karan/oa/marin/.env'); configure_coreweave_s3()
    credentials = {k:require_secret(k) for k in ['CW_KEY_ID','CW_KEY_SECRET']}
    from experiments.post_training.identity_path_b.a12_export_recovery import checked_original_training, binding as committed_binding
    actual, original_event = checked_original_training()
    assert original_event.payload() == item['original_event'] and actual['metadata']['step'] == 17
    for key in ['source_receipt','source_objects']:
        entry = item['training'][key]; assert sha(get(entry['path'])) == entry['sha256']
    f,k = filesystem_for(root); assert not f.exists(k) and not f.find(k)
    f,k = filesystem_for(item['original_event']['hf_export']); assert not f.exists(k) and not f.find(k), 'Do not repeat an existing valid HF export'
    old = json.loads((P/'ablation-a12-a13-r1/A12/actual-terminal-attempts-and-artifact-coverage-margay-r1.json').read_bytes())
    assert old['all_owned_jobs_terminal'] is True
    assert sha(create_workspace_zip(C)) == ZIP
    spec = importlib.util.spec_from_file_location('sealed_A12_export_only', D/'prepare_request.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); kwargs = mod.request(credentials)
    inspect.signature(IrisClient.submit).bind(object(), **kwargs)
    fn, positional, kw = kwargs['entrypoint'].resolve(); inspect.signature(fn).bind(*positional, **kw)
    assert not positional and fn.__name__ == 'export_only_worker' and kw == {'runtime_bytes':runtime} and kwargs['entrypoint'].workdir_files
    assert kwargs['name'] == item['job_name'] and kwargs['resources'].device is None
    assert kwargs['priority_band'] == job_pb2.PRIORITY_BAND_PRODUCTION and kwargs['max_retries_failure'] == kwargs['max_retries_preemption'] == 0 and kwargs['timeout'].to_seconds() == 176400
    assert 'MARIN_PREFIX' not in kwargs['environment'].env_vars
    assert committed_binding('A12')['root'] == root and committed_binding('A12')['event'] == event
    assert kwargs['environment'].env_vars['JAX_PLATFORMS'] == 'cpu'
    expected_job = '/karan/'+item['job_name']
    with connect_controller(cluster_name='marin') as co:
        with IrisClient.remote(co.url, workspace=C, credentials=co.credentials) as client:
            assert not client.list_jobs(prefix=expected_job)
            prior_log=Path('/mnt/storage/karan-scratch/margay-A12-export-only-recovery-launch-r1.log')
            clarification=D/'operator-MARIN_PREFIX-preflight-zero-call-clarification-margay-r1.json'
            prior={'utc':utc(),'state':'LOCAL_EXECUTOR_GUARD_CORRECTED_ZERO_APPLICATION_CALLS','prior_log':str(prior_log),'prior_log_sha256':sha(prior_log.read_bytes()),'cause':'Executor wrongly required SWE-style MARIN_PREFIX in a12 request(credentials). Actual reviewed owner uses immutable committed binding/root/event and CPU precision env, not prepare global MARIN_PREFIX. No source/receipt/plan/native ZIP or scientific change.','prior_failure_before_client_connection_or_exclusive_launch_journal':True,'actual_expected_job_absent':expected_job,'new_output_namespace_absent':root,'application_submit_calls_before_correction':0,'actual_namespace_guard':'Committed binding exact root+event plus actual callable/runtime and CPU env; no MARIN_PREFIX field required'}
            with clarification.open('x') as h:json.dump(prior,h,indent=2);h.write('\n')
            r = {'state':'PRE_PROBE','utc':utc(),'arm':'A12','mode':'EXPORT_ONLY_NO_TRAINING','physical_export_recovery_number':1,'authority':'DIRECTJerboa ONE production export-only recovery + cb426 narrowCLEAR; general finished-train/failed-export rule','source_pin':PIN,'scientific_training_pin':TRAIN,'sealed_files':sealed,'local_native_bundle_sha256':ZIP,'orders_sha256_submission_only':sha(orders),'expected_job':expected_job,'run_root':root,'target_HF':event['hf_export'],'original_protected_native17':event['native_checkpoint'],'original_terminal_evidence_sha256':'8ad9c712ff025a9a064cc69eae92d146bbce18bbc2d73aaace88eca57695ccf5','actual_trained18_source_proof_sha256':plan['source_proof_sha256'],'retire_ONLY_original_native17_AFTER_fresh_full502audit':True,'no_retraining_or_resume':True,'required_callable_workdir_serialization_preserved':True,'CPU_owner_AND_owned_export_priority':'production','actual_namespace_guard':'Exact committed binding root+event, callable/runtime and scientific metadata; SDK uses CPU env without MARIN_PREFIX, unlike SWE prepare API','application_submit_calls':0,'credentials_in_receipt':False,'actual_HF_readiness_NOT_claimed':True}
            def save():
                with out.open('w') as h: json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
            with out.open('x') as h: json.dump(r,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
            body = secrets.token_bytes(32); key = 'tmp/ttl=14d/snowball-identity/path-b/A12-export-recovery-submit-probes/'+secrets.token_hex(8)+'.bin'
            r['write_probe'] = {'uri':'s3://marin-rno2a/'+key,'bytes':32,'sha256':sha(body),'state':'PUT_CALLED'}; save()
            s3 = get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
            try:
                s3.put_object(Bucket='marin-rno2a',Key=key,Body=body); stream=s3.get_object(Bucket='marin-rno2a',Key=key)['Body']
                try: assert stream.read() == body
                finally: stream.close()
            finally: s3.close()
            r['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS'; save(); assert N.read_bytes() == orders
            for name,h in sealed.items(): assert sha((D/name).read_bytes()) == h
            r.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc()); save()
            try: job = client.submit(**kwargs)
            except Exception as exc:
                r.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(exc).__name__);save();raise RuntimeError('Reconcile root read-only; never blind retry') from None
            r.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert r['job'] == expected_job
            log=(D/'completion-notifier.log').open('ab')
            proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',r['job'],'--timeout','180000','--out',str(D/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();r['one_completion_notifier_pid']=proc.pid;save()
            try:
                q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=r['job'])).request
                cs=[MessageToDict(c,preserving_proto_field_name=True) for c in q.constraints]
                r['actual_native_request']={'bundle_id':q.bundle_id,'priority':job_pb2.PriorityBand.Name(q.priority_band),'constraints':cs,'resources':MessageToDict(q.resources,preserving_proto_field_name=True),'timeout_ms':q.timeout.milliseconds,'failure_retries':q.max_retries_failure,'preemption_retries':q.max_retries_preemption}
                r['actual_native_binding_PASS']=q.bundle_id==ZIP and q.priority_band==job_pb2.PRIORITY_BAND_PRODUCTION and 'cw-rno2a' in json.dumps(cs) and q.timeout.milliseconds==176400000 and q.max_retries_failure==q.max_retries_preemption==0 and not q.resources.device.ListFields()
                save()
            except Exception as exc: r['native_readback_error_type']=type(exc).__name__;save()
    print('A12_EXPORT_ONLY_RECOVERY_SUBMITTED',str(out),sha(out.read_bytes()),r['job'],'NATIVE_PASS',r.get('actual_native_binding_PASS'),flush=True)
    assert r.get('actual_native_binding_PASS') is True,'Already submitted: do not repeat consumed call'
if __name__=='__main__': main()
