"""One BFCL per auditedr5HF, HF-only and independent of native slot release. Not a watcher."""
import argparse,datetime,hashlib,json,os,re,subprocess,sys
from pathlib import Path
from dotenv import load_dotenv
from iris.client.client import IrisClient
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration
N=Path('/home/karan/Dropbox/marin-notes/snowball-identity');P=N/'path-b';C=Path('/mnt/storage/karan-scratch/marin-held-bfcl-plumbing');PIN='4bc3742fe7726188ed001daecf805d97cd191a20'
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-name',required=True);parser.add_argument('--updates',required=True,type=int);parser.add_argument('--orders-sha256',required=True);args=parser.parse_args()
 os.chdir(C);assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 vr=(P/'r5-weight1-r1/science-verdict-marten.json').read_bytes();assert sha(vr)=='0368fa851f8d84f8f2d16538f9e1df1c4772e64e6d3f8b8959be9faf639f8f8c';verdict=json.loads(vr);raw=(P/'r5-weight1-r1/launch-plan.json').read_bytes();assert sha(raw)==verdict['plan_sha256'];plan=json.loads(raw);a=next(a for a in plan['arms'] if a['root_name']==args.root_name);cp=next(x for x in a['checkpoints'] if x['completed_updates']==args.updates);assert plan['code_pin']==PIN
 label=('bbase' if a['variant']=='B-base' else 'btraces')+'-'+a['arm'];version=f'pathb-r5-{label}-u{args.updates}-4bc3742fe772';D=P/'r5-checkpoint-bfcl-r1'/args.root_name/f'updates-{args.updates}';OUT=D/'launch.json';assert not OUT.exists(),'ConsumedBFCLjournal; no automaticretry'
 load_dotenv('/home/karan/oa/marin/.env');configure_coreweave_s3();read={}
 for name in ('path-b-checkpoint-event.json','path-b-storage-audit.json'):
  f,k=filesystem_for(cp['hf_export']+'/'+name)
  if not f.exists(k):print(json.dumps({'state':'NO_AUDITED_HF_NO_SUBMIT','root':args.root_name,'updates':args.updates,'missing':name,'native_slot_operations':0}));return
  read[name]=f.cat_file(k)
 event=json.loads(read['path-b-checkpoint-event.json']);audit=json.loads(read['path-b-storage-audit.json'])
 for key,value in {'arm':a['arm'],'completed_updates':args.updates,'native_checkpoint':cp['native_checkpoint'],'hf_export':cp['hf_export'],'eval_scratch':cp['eval_scratch'],'exposure':cp['cumulative_actual_exposure'],'training_pin':PIN,'packed_data_sha256':a['training_kwargs']['packed_sha256']}.items():assert event[key]==value,key
 assert audit['schema']=='path_b_storage_v1' and audit['source_receipt_sha256']==plan['source_receipt']['sha256'] and audit['frozen_native_tensors']==104 and audit['trained_master_cast_tensors']==398 and len(audit['tensors'])==502
 assert sum(bool(x['frozen_source']) for x in audit['tensors'].values())==104 and all(re.fullmatch('[0-9a-f]{64}',x['export_sha256']) for x in audit['tensors'].values())
 science=(P/'full-screen-execution-r1/execution-packet.json').read_bytes();assert sha(science)=='13b058213e399e402d36a3f1c5612dc885b4345381939e36aaec351268deebb2';template=next(c for c in json.loads(science)['candidates'] if c['label']=='pathb-base25-u52-r1');b=template['bfcl']
 for key,h in [('model_config','model_config_sha256'),('policy_wrapper','policy_wrapper_sha256')]:assert sha(Path(b[key]).read_bytes())==b[h]
 artifact=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/r5-checkpoint-bfcl-r1/{label}/updates-{args.updates}';records=artifact+'/records';f,k=filesystem_for(records);assert not f.exists(k) and not f.find(k),'BFCLrecordsnamespace notfresh'
 D.mkdir(parents=True,exist_ok=True)
 for name,payload in read.items():
  path=D/name
  if path.exists():assert path.read_bytes()==payload
  else:
   with path.open('xb') as h:h.write(payload)
 model=D/'model.yaml';text=Path(b['model_config']).read_text();old_name=text.splitlines()[0];text=text.replace(old_name,'name: snowball-'+version,1).replace('location: '+template['model_path'],'location: '+cp['hf_export'],1)
 if model.exists():assert model.read_text()==text
 else:
  with model.open('x') as h:h.write(text)
 import yaml
 old=yaml.safe_load(Path(b['model_config']).read_bytes());new=yaml.safe_load(model.read_bytes());assert {k:v for k,v in old.items() if k not in ('name','location')}=={k:v for k,v in new.items() if k not in ('name','location')}
 orders_path=N/'t310/current-orders.md';orders=orders_path.read_bytes();assert sha(orders)==args.orders_sha256
 os.environ['MARIN_PREFIX']=artifact+'/native';os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS',None)
 argv=list(b['argv']);argv[argv.index('--model-config')+1]=str(model);argv[argv.index('--version')+1]=version;argv[argv.index('--records-prefix')+1]=records;argv[argv.index('--description')+1]=f'R5 {label}@{args.updates} auditedHF checkpoint BFCL; native slots independent'
 record={'state':'PRE_SUBMIT','utc':utc(),'authority':'Jerboa explicitlyBFCL on EACH r5auditedHF, firstcheckpointpriority; unchangedclearedmethod/modelpath-only plus freshidentities','root':args.root_name,'updates':args.updates,'hf_export':cp['hf_export'],'event_sha256':sha(read['path-b-checkpoint-event.json']),'audit_sha256':sha(read['path-b-storage-audit.json']),'source_science_packet_sha256':sha(science),'model_yaml_sha256':sha(model.read_bytes()),'records_prefix':records,'orders_sha256_submission_only':sha(orders),'application_submit_calls':0,'native_checkpoint_reads_leases_deletes':0,'native_slot_release_never_gated':True,'no_automatic_retry':True,'outer_CPU_lifetime_seconds':49*3600,'native_serve_lifetime_seconds':12*3600}
 def save():
  with OUT.open('w') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 with OUT.open('x') as h:json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 original=IrisClient.submit
 def submit(client,*pos,**kwargs):
  assert record['application_submit_calls']==0 and orders_path.read_bytes()==orders;kwargs['timeout']=Duration.from_hours(49);assert kwargs['max_retries_failure']==kwargs['max_retries_preemption']==0;expected='/karan/'+kwargs['name'];assert not client.list_jobs(prefix=expected);record.update(state='SUBMISSION_CALLED',expected_job=expected,application_submit_calls=1,submit_called_utc=utc());save()
  try:job=original(client,*pos,**kwargs)
  except Exception as error:record.update(state='SUBMISSION_UNCERTAIN_NO_RETRY',error_type=type(error).__name__);save();raise RuntimeError('Reconcile exactnamespace,noautomaticretry') from None
  record.update(state='SUBMITTED',job=str(job.job_id),returned_utc=utc());assert record['job']==expected;save();return job
 IrisClient.submit=submit
 try:
  from experiments.evaluation import held_bfcl
  sys.argv=[argv[2],*argv[3:]];held_bfcl.main()
 finally:IrisClient.submit=original
 assert record['state']=='SUBMITTED';log=(D/'notifier.log').open('ab');proc=subprocess.Popen([sys.executable,'/mnt/storage/karan-scratch/margay-notify-job-terminal.py','--job',record['job'],'--timeout','180000','--out',str(D/'completion.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();record['single_completion_notifier_pid']=proc.pid;save();print(json.dumps({'receipt':str(OUT),'receipt_sha256':sha(OUT.read_bytes()),**record}),flush=True)
if __name__=='__main__':main()
