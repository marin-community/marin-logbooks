"""Submit ONE admitted CPU merge binding, no source weights loaded locally or retry."""
import argparse,datetime,hashlib,importlib.util,inspect,json,os,secrets,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from botocore.session import get_session
from botocore.config import Config
from google.protobuf.json_format import MessageToDict
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import controller_pb2,job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3,s3_endpoint
from rigging.filesystem.cluster_config import StoreType
from rigging.secrets import require_secret
P=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');D=P/'merge-sweep-r3';C=Path('/home/karan/oa/marin-path-b-merge-sweep-r3');N=P.parent/'t310/current-orders.md';PIN='462bba9353b562f841d095a6265dac47d5cafafe';ZIP='7b02d28f95cff1da58b14366501a171c407951630ed833e09e4a636bad996782';CLEAR='8e19f6bf677f3b87b568a1c797c8b9764422b089f9c89a9ab82d9f79bd1fb8fb';RECEIPT='08bdb912d0d69a34f53efb83e5d51b79d136c0e277bcb9552472cb4b05b5e516';PLAN='3ef26b76dbcd397e1682ae017ff9ea6528d2f4a89103c76fae0e3dde5d79ae42';ALPHAS={'M25':.25,'M49':.49,'M51':.51,'M75':.75}
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(uri):
 f,k=filesystem_for(uri);return f.cat_file(k)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',choices=list(ALPHAS),required=True);ap.add_argument('--orders-sha256',required=True);a=ap.parse_args();os.chdir(C);sd=D/a.label;sd.mkdir(exist_ok=True);out=sd/'actual-CPU-merge-launch-margay-r1.json';assert not out.exists(),'Consumed journal: reconcile read-only, never blindly repeat';assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==PIN and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();required={'binding-review-marten-r1.json':CLEAR,'review-receipt.json':RECEIPT,'launch-plan.json':PLAN}
 for name,h in required.items():assert sha((D/name).read_bytes())==h
 review=json.loads((D/'binding-review-marten-r1.json').read_bytes());assert review['verdict']=='CLEAR_TRANSPORT_ONLY_M25_SETUP_REPLACEMENT1_AND_THREE_INITIAL_CPU_MERGES'and review['source_pin']==PIN and review['receipt_sha256']==RECEIPT and review['plan_sha256']==PLAN and review['native_bundle_sha256']==ZIP;proof=review['independent_proof'];assert sha(Path(proof['path']).read_bytes())==proof['sha256']=='b20c32f2e230ba8167508b67359aa20421941e91329b13004bd2eaf8fb46ded8'
 for path,r in json.loads((D/'review-receipt.json').read_bytes())['files'].items():
  b=Path(path).read_bytes();assert len(b)==r['bytes']and sha(b)==r['sha256']
 plan=json.loads((D/'launch-plan.json').read_bytes());assert{r['label']:r['alpha']for r in plan['roots']}==ALPHAS;target=next(r for r in plan['roots']if r['label']==a.label)
 for field in['preregistration','frozen_restoration_clarification','native_proof']:assert sha(Path(plan[field]['path']).read_bytes())==plan[field]['sha256']
 orders=N.read_bytes();assert sha(orders)==a.orders_sha256 and b'0.49, 0.51'in orders and b'MERGE SWEEP'in orders;load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();credentials={k:require_secret(k)for k in['CW_KEY_ID','CW_KEY_SECRET']};A=C/'experiments/weight_merging/path_b_sweep/artifacts';recipe_bytes=(A/(a.label+'.json')).read_bytes();recipe=json.loads(recipe_bytes);assert recipe['parameters']=={'coefficients':[ALPHAS[a.label]],'density':1,'method':'average','scale':1,'seed':0}and not recipe['tensor_coefficients']and not recipe['row_overrides']and len(recipe['donors'])==1;assert recipe['anchor']['path']=='s3://marin-rno2a/marin/checkpoints/karan/snowball-identity-pathb/A3-u36'and recipe['donors'][0]['path']=='s3://marin-rno2a/marin/checkpoints/karan/snowball-identity-pathb/A8a-u36';assert len(recipe['preserve_rows'])==106 and recipe['output']==target['output'];assert recipe['preserve_rows']['model.embed_tokens.weight']==recipe['preserve_rows']['lm_head.weight']==list(range(128000,128256))
 # Only small source metadata is read locally; full parent/output SHA belongs to RNO CPU worker.
 parent_metadata=[]
 for label in['A3','A8a']:
  p=json.loads((A/(label+'-preservation.json')).read_bytes());root=p['destination'].rstrip('/')
  for r in p['objects']:
   if r['relative']in['config.json','tokenizer_config.json','chat_template.jinja','model.safetensors.index.json']:
    b=get(root+'/'+r['relative']);assert len(b)==r['bytes']and sha(b)==r['destination_sha256'];parent_metadata.append({'parent':label,'relative':r['relative'],'sha256':sha(b),'bytes':len(b)})
 root=recipe['output'].rsplit('/hf',1)[0];fs,key=filesystem_for(root);assert not fs.exists(key)and not fs.find(key),'Fresh outputs required; no overwrite';assert sha(create_workspace_zip(C))==ZIP;spec=importlib.util.spec_from_file_location('sealed_merge_cpu',D/'prepare_request.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kwargs=mod.request(a.label,credentials);inspect.signature(IrisClient.submit).bind(object(),**kwargs);fn,args,kw=kwargs['entrypoint'].resolve();inspect.signature(fn).bind(*args,**kw);assert not args and kw=={'label':a.label,'code_revision':PIN}and kwargs['entrypoint'].workdir_files;res=kwargs['resources'];assert res.cpu==16 and res.memory=='64g'and res.disk=='40g'and res.device is None;assert kwargs['replicas']==1 and kwargs['timeout'].to_seconds()==43200 and kwargs['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION and kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;assert kwargs['name']==target['name'];constraints=[MessageToDict(c.to_proto(),preserving_proto_field_name=True)for c in kwargs['constraints']];assert'cw-rno2a'in json.dumps(constraints);expectedjob='/karan/'+target['name']
 with connect_controller(cluster_name='marin')as co:
  with IrisClient.remote(co.url,workspace=C,credentials=co.credentials)as client:
   assert not client.list_jobs(prefix=expectedjob);record={'state':'PRE_PROBE','utc':utc(),'label':a.label,'alpha':ALPHAS[a.label],'logical_setup_replacement_number':target['logical_setup_replacement_number'],'authority':'Jerboa ONEtransport-onlyM25setupR1 plusother3initials + 8e19CLEAR; NO M50; technicalr3 notthirdattempt','source_pin':PIN,'sealed_files':required,'orders_sha256_submission_only':sha(orders),'expected_job':expectedjob,'output':recipe['output'],'local_native_bundle_sha256':ZIP,'recipe_sha256':sha(recipe_bytes),'parent_metadata_readback':parent_metadata,'required_callable_workdir_serialization_preserved':True,'parent_model_local_weight_reads':0,'provenance_kind':'linear-cpu-merge','no_fake_training_event_or_mastercast_audit':True,'actual_merge_admission_required_before_ANY_eval':True,'resource_policy':'CPU16/64g/40g/12h/RNO/PRODUCTION0-0','application_submit_calls':0,'automatic_retry':False,'credentials_in_receipt':False,'parentS3deletion':'HOLD'}
   def save():
    with out.open('w')as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   with out.open('x')as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   probe=secrets.token_bytes(32);k='tmp/ttl=14d/snowball-identity/path-b/merge-submit-probes/'+a.label+'-'+secrets.token_hex(8)+'.bin';record['write_probe']={'uri':'s3://marin-rno2a/'+k,'bytes':32,'sha256':sha(probe),'state':'PUT_CALLED'};save();s3=get_session().create_client('s3',endpoint_url=s3_endpoint(StoreType.COREWEAVE),region_name='auto',aws_access_key_id=credentials['CW_KEY_ID'],aws_secret_access_key=credentials['CW_KEY_SECRET'],config=Config(s3={'addressing_style':'virtual'},connect_timeout=10,read_timeout=20,retries={'total_max_attempts':1,'mode':'standard'},request_checksum_calculation='when_required',response_checksum_validation='when_required'))
   try:
    s3.put_object(Bucket='marin-rno2a',Key=k,Body=probe);stream=s3.get_object(Bucket='marin-rno2a',Key=k)['Body']
    try:assert stream.read()==probe
    finally:stream.close()
   finally:s3.close()
   record['write_probe']['state']='PUT_FULLGET_BYTE_EQUAL_PASS';save();assert N.read_bytes()==orders
   for name,h in required.items():assert sha((D/name).read_bytes())==h
   record.update(state='SUBMISSION_CALLED',application_submit_calls=1,called_utc=utc());save()
   try:job=client.submit(**kwargs)
   except Exception as exc:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(exc).__name__);save();raise RuntimeError('Read-only reconcile; no blind retry')from None
   record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());save();assert record['job']==expectedjob;log=(sd/'completion-notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','45000','--out',str(sd/'completion-native.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['one_completion_notifier_pid']=proc.pid;save()
   try:
    q=client._cluster_client._client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=record['job'])).request;cs=[MessageToDict(c,preserving_proto_field_name=True)for c in q.constraints];record['actual_native_request']={'bundle_id':q.bundle_id,'priority':job_pb2.PriorityBand.Name(q.priority_band),'constraints':cs,'resources':MessageToDict(q.resources,preserving_proto_field_name=True),'timeout_ms':q.timeout.milliseconds,'failure_retries':q.max_retries_failure,'preemption_retries':q.max_retries_preemption};record['actual_native_binding_PASS']=q.bundle_id==ZIP and q.priority_band==job_pb2.PRIORITY_BAND_PRODUCTION and'cw-rno2a'in json.dumps(cs)and q.max_retries_failure==q.max_retries_preemption==0 and q.timeout.milliseconds==43200000 and q.resources.cpu_millicores==16000 and q.resources.memory_bytes==res.to_proto().memory_bytes and q.resources.disk_bytes==res.to_proto().disk_bytes and not q.resources.device.ListFields();save()
   except Exception as exc:record['native_readback_error_type']=type(exc).__name__;save()
 print(json.dumps({'receipt':str(out),'sha256':sha(out.read_bytes()),'job':record['job'],'label':a.label,'one_notifier':proc.pid,'actual_native_binding_PASS':record.get('actual_native_binding_PASS')}),flush=True);assert record.get('actual_native_binding_PASS')is True,'Already submitted: reconcile; never repeat consumed call'
if __name__=='__main__':main()
