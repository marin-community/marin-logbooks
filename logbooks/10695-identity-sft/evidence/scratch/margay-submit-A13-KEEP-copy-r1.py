"""ONE A13 KEEP-fallbacks HF preservation job: reuses the UNCHANGED preserve_hf from
margay-preserve-HF-fallbacks-r2.py with vizcacha's sealed A13-copy-spec. Same request shape as main().
Read-only preconditions verified; no source deletion; no overwrite; no workflow retry."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,subprocess,sys
from pathlib import Path
def sha(b):return hashlib.sha256(b).hexdigest()
def load_runner():
    spec=importlib.util.spec_from_file_location('preserve_hf_runner','/mnt/storage/karan-scratch/margay-preserve-HF-fallbacks-r2.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--orders-sha256',required=True);a=ap.parse_args()
    from dotenv import load_dotenv
    from iris.cli.connect import connect_controller
    from iris.client.client import IrisClient
    from iris.cluster.types import Entrypoint,EnvironmentSpec,ResourceSpec
    from iris.rpc import controller_pb2,job_pb2
    from fray.iris_backend import convert_constraints
    from fray.types import ResourceConfig
    from rigging.filesystem.buckets import filesystem_for
    from rigging.filesystem.s3_compat import configure_coreweave_s3
    from rigging.secrets import require_secret,SecretBehavior
    from rigging.timing import Duration
    from google.protobuf.json_format import MessageToDict
    P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');N=P.parent/'t310/current-orders.md'
    spec_path=P/'keep-fallbacks-a13-r1/A13-copy-spec.json';spec_raw=spec_path.read_bytes()
    assert sha(spec_raw)=='d1e71b3059ed5f4102ac6821d0b5b61db72d0513f48ef9e249e72f018ac544ee','sealed A13 spec changed'
    spec=json.loads(spec_raw);spec_core={k:v for k,v in spec.items()if k!='spec_sha256'}
    assert sha(json.dumps(spec_core,sort_keys=True).encode())==spec['spec_sha256']=='060071dafba8b9e1a1dab4bff309f69a696b9420b38216a61e3c5558f37f21a4'
    pinned=spec['pinned_metadata']
    assert pinned['path-b-storage-audit.json']=='4bcfe2c44bad2214129d27325e7dfcc427f63b8d86553a643bc92e81809202f2'
    assert pinned['path-b-checkpoint-event.json']=='998b91183e8fb3ad70c5502015ff0765c50b82d40f02424a77f7ec2a7fa456c9'
    assert spec['arm']=='A13' and len(spec['objects'])==56 and sum(o['bytes']for o in spec['objects'])==134175473290
    assert spec['destination']=='s3://marin-rno2a/marin/checkpoints/karan/snowball-identity-pathb/A13-u36'
    D=P/'keep-fallbacks-a13-r1';C=Path('/home/karan/oa/marin-path-b-a9-72');os.chdir(C)
    orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'KEEP FALLBACKS' in orders
    out=D/'A13-launch-margay-r1.json';assert not out.exists(),'Consumed launch journal; read-only reconciliation only'
    load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3()
    fs,src=filesystem_for(spec['source']);src=fs._strip_protocol(src);_,dst=filesystem_for(spec['destination']);dst=fs._strip_protocol(dst)
    # fresh destination + progress; source inventory exactly equals the sealed object list
    assert not fs.find(dst),'A13 destination already exists; never overwrite'
    assert not fs.find('marin-rno2a/'+spec['progress_key']),'A13 progress prefix already exists; no workflow rerun'
    live={}
    for key,info in sorted(fs.find(src,detail=True).items()):
        if info.get('type')!='file':continue
        rel=fs._strip_protocol(key)[len(src)+1:];live[rel]={'relative':rel,'bytes':int(info['size']),'etag':info['ETag']}
    assert live=={o['relative']:o for o in spec['objects']},'A13 source inventory changed since the spec was sealed'
    for name,digest in pinned.items():assert sha(fs.cat_file(src+'/'+name))==digest,name
    credentials={k:require_secret(k) for k in['CW_KEY_ID','CW_KEY_SECRET']};credentials.update(AWS_ACCESS_KEY_ID=credentials['CW_KEY_ID'],AWS_SECRET_ACCESS_KEY=credentials['CW_KEY_SECRET'])
    runner=load_runner();entrypoint=Entrypoint.from_callable(runner.preserve_hf,spec=spec);assert entrypoint.workdir_files;fn,args,kw=entrypoint.resolve();inspect.signature(fn).bind(*args,**kw)
    name='pathb-keep-hf-a13-r1-20261031'
    kwargs=dict(name=name,entrypoint=entrypoint,resources=ResourceSpec(cpu=4,memory='16g',disk='20g'),constraints=convert_constraints(ResourceConfig.with_cpu(cpu=4,ram='16g',disk='20g',target_cluster='cw-rno2a')),environment=EnvironmentSpec(extras=['cpu'],pip_packages=['boto3==1.41.5','botocore==1.41.5','s3transfer==0.15.0'],secrets=credentials,secret_behavior=SecretBehavior.ISOLATE,env_vars={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','JAX_PLATFORMS':'cpu'}),replicas=1,timeout=Duration.from_seconds(43200),max_retries_failure=0,max_retries_preemption=0,priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE);inspect.signature(IrisClient.submit).bind(object(),**kwargs)
    with connect_controller(cluster_name='marin')as co:
        with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
            expected='/karan/'+name;assert not client.list_jobs(prefix=expected)
            record={'state':'PRE_SUBMIT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arm':'A13','authority':"Jerboa-approved vizcacha 3-step A13 archive route, step 1 (KEEP copy); owner KEEP-fallback path; no new science gate",'spec_file':str(spec_path),'spec_file_sha256':sha(spec_raw),'spec_sha256':spec['spec_sha256'],'orders_sha256_submission_only':sha(orders),'script_sha256':sha(Path(__file__).read_bytes()),'expected_job':expected,'source':spec['source'],'destination':spec['destination'],'progress':'s3://marin-rno2a/'+spec['progress_key'],'objects':len(spec['objects']),'bytes':sum(o['bytes']for o in spec['objects']),'pinned_audit_sha256':pinned['path-b-storage-audit.json'],'pinned_event_sha256':pinned['path-b-checkpoint-event.json'],'application_submit_calls':0,'fresh_destination_and_progress_verified':True,'source_inventory_equals_sealed_spec':True,'no_model_GPU_native_checkpoint_delete_or_reexport':True,'automatic_workflow_retry':False,'credentials_in_receipt':False}
            def save():
                with out.open('w')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
            with out.open('x')as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
            assert N.read_bytes()==orders;record.update(state='SUBMISSION_CALLED',application_submit_calls=1);save()
            try:job=client.submit(**kwargs)
            except Exception as error:record.update(state='UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Read-only reconcile; no blind retry')from None
            record.update(state='SUBMITTED',job=str(job.job_id));save();assert record['job']==expected
            log=(D/'A13-completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','45000','--out',str(D/'A13-completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_notifier_pid']=proc.pid;save()
            q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request
            record['actual_native_policy']={'bundle_id':q.bundle_id,'constraints':[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints],'failure_retries':q.max_retries_failure,'preemption_retries':q.max_retries_preemption,'priority':job_pb2.PriorityBand.Name(q.priority_band),'timeout_ms':q.timeout.milliseconds};save()
            print(json.dumps({'job':record['job'],'receipt':str(out),'sha256':sha(out.read_bytes()),'one_notifier_pid':proc.pid,'objects':len(spec['objects']),'bytes':record['bytes']}),flush=True)
if __name__=='__main__':main()
