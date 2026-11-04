"""One-shot BFCL for one actually audited ablation finalHF; NOT watcher/native reader."""
import argparse,datetime,hashlib,json,os,re,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
N=Path('/home/karan/Dropbox/marin-notes/snowball-identity');D=N/'path-b/bfcl-ablation-r1';C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing');PIN='6ac2c53d061156ad22be53b332b7bd677a856290'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['A0','A1','A2','A3','A4'],required=True);p.add_argument('--orders-sha256',required=True);args=p.parse_args();os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501';vr=(D/'science-verdict-marten.json').read_bytes();assert sha(vr)=='886302a2d113678d7c29157f836e7da7c63b3d0e9f575f19001588bc3c235881';v=json.loads(vr);raw=(D/'launch-plan.json').read_bytes();assert sha(raw)==v['plan_sha256'] and v['submit_authorized'] and v['decision']=='CLEAR';plan=json.loads(raw);assert plan['code_pin']==PIN;a=next(x for x in plan['arms'] if x['arm']==args.arm);admit=next(x for x in v['arms'] if x['arm']==args.arm);assert a['training_kwargs']==admit['training_kwargs'];event_expected=a['final_checkpoint'];hf=event_expected['hf_export'];outdir=D/args.arm;OUT=outdir/'bfcl-launch.json';assert not OUT.exists(),'Consumed one-shotBFCL journal; do not rerun';load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();blobs={}
 for name in ('path-b-checkpoint-event.json','path-b-storage-audit.json'):
  fs,k=filesystem_for(hf+'/'+name)
  if not fs.exists(k):print(json.dumps({'state':'NO_AUDITED_HF_NO_SUBMIT','arm':args.arm,'missing':name,'native_checkpoint_reads_leases_deletes':0}));return
  blobs[name]=fs.cat_file(k)
 event=json.loads(blobs['path-b-checkpoint-event.json']);audit=json.loads(blobs['path-b-storage-audit.json'])
 for key,value in event_expected.items():assert event[key]==value,key
 assert event['training_pin']==PIN and event['completed_updates']==36 and audit['schema']=='path_b_storage_v1' and audit['source_receipt_sha256']==plan['source_receipt']['sha256'];assert audit['frozen_native_tensors']==a['frozen_native_tensors']==admit['frozen_native_tensors'] and audit['trained_master_cast_tensors']==a['trained_master_cast_tensors']==admit['trained_master_cast_tensors'] and len(audit['tensors'])==502;assert sum(bool(t['frozen_source']) for t in audit['tensors'].values())==a['frozen_native_tensors'] and all(re.fullmatch('[0-9a-f]{64}',t['export_sha256']) for t in audit['tensors'].values())
 b=a['bfcl'];assert b['workspace']==str(C) and b['workspace_pin']=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 for key,h in [('model_config','model_config_sha256'),('policy_wrapper','policy_wrapper_sha256')]:assert sha(Path(b[key]).read_bytes())==b[h]
 assert sha(Path(b['model_config']).read_bytes())==admit['bfcl_model_config_sha256'];f,k=filesystem_for(b['records_prefix']);assert not f.exists(k) and not f.find(k),'Records prefix notfresh';orders_path=N/'t310/current-orders.md';orders=orders_path.read_bytes();assert sha(orders)==args.orders_sha256;outdir.mkdir(exist_ok=True)
 for name,payload in blobs.items():
  path=outdir/name
  if path.exists():assert path.read_bytes()==payload
  else:
   with path.open('xb') as h:h.write(payload)
 os.environ['MARIN_PREFIX']=b['MARIN_PREFIX'];os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None);record={'state':'PRE_SUBMIT','utc':utc(),'arm':args.arm,'training_pin':PIN,'hf_export':hf,'science_verdict_sha256':sha(vr),'plan_sha256':sha(raw),'event_sha256':sha(blobs['path-b-checkpoint-event.json']),'audit_sha256':sha(blobs['path-b-storage-audit.json']),'model_yaml_sha256':b['model_config_sha256'],'records_prefix':b['records_prefix'],'authority':'Jerboa/Marten combined fivearm36update ablation; one unchangedcabaBFCL per audited finalHF','orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'native_checkpoint_reads_leases_deletes':0,'no_automatic_retry':True,'outer_CPU_lifetime_seconds':176400,'native_serve_lifetime_seconds':43200}
 def save():
  with OUT.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with OUT.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 original=IrisClient.submit
 def submit(client,*pos,**kwargs):
  assert record['application_submit_calls']==0 and orders_path.read_bytes()==orders;kwargs['timeout']=Duration.from_hours(49);assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;expected='/karan/'+kwargs['name'];assert not client.list_jobs(prefix=expected);record.update(state='SUBMISSION_CALLED',expected_job=expected,application_submit_calls=1,called_utc=utc());save()
  try:job=original(client,*pos,**kwargs)
  except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile exact namespace readonly; never automatic retry') from None
  record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();return job
 IrisClient.submit=submit
 try:
  from experiments.evaluation import held_bfcl
  argv=b['argv'];sys.argv=[argv[2],*argv[3:]];held_bfcl.main()
 finally:IrisClient.submit=original
 assert record['state']=='SUBMITTED';log=(outdir/'completion.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(outdir/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['single_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(OUT),'sha256':sha(OUT.read_bytes()),**record}),flush=True)
if __name__=='__main__':main()
