from pathlib import Path
import copy,hashlib,inspect,io,json,subprocess,zipfile
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import job_pb2
from experiments.post_training.identity_path_b.ablation_launch import submission,workflow_worker
from experiments.post_training.identity_path_b.ablation import train_worker,export_worker
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
P=Path(__file__).resolve().parent

def main():
 plan=json.loads((P/'launch-plan.json').read_text());arm=plan['arm'];before=copy.deepcopy(plan['original_training_kwargs']);after=copy.deepcopy(arm['training_kwargs']);assert {k for k in before if before[k]!=after[k]}=={'run_id','run_root'};assert set(before)==set(after)
 runtime=Path(plan['runtime']['local']).read_bytes();assert hashlib.sha256(runtime).hexdigest()==plan['runtime']['sha256'];after['runtime_bytes']=runtime
 request=submission(name=arm['root_name'],training=after,credentials=dict(CW_KEY_ID='fixture-only',CW_KEY_SECRET='fixture-only'));inspect.signature(IrisClient.submit).bind(None,**request);fn,args,kwargs=request['entrypoint'].resolve();assert fn is workflow_worker and kwargs=={'training':after};inspect.signature(fn).bind(*args,**kwargs);inspect.signature(train_worker).bind(**after)
 event=CheckpointEvent(**arm['final_checkpoint']);assert event.completed_updates==36 and event.native_checkpoint.endswith('/step-35') and event.training_pin==plan['code_pin'];event.payload();inspect.signature(export_worker).bind(event=event,**{k:after[k]for k in ['run_id','runtime_bytes','runtime_sha256','source_receipt','source_objects']})
 assert request['priority_band']==job_pb2.PRIORITY_BAND_INTERACTIVE and request['max_retries_failure']==request['max_retries_preemption']==0 and request['timeout'].to_seconds()==176400
 root=Path(plan['workspace']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
 bundle=create_workspace_zip(root);assert hashlib.sha256(bundle).hexdigest()==plan['native_bundle_sha256']
 (P/'native-unchanged-source-and-namespace-proof.json').write_text(json.dumps({'code_pin':plan['code_pin'],'bundle_sha256':hashlib.sha256(bundle).hexdigest(),'bundle_bytes':len(bundle),'only_training_deltas':['run_id','run_root'],'new_root_name':arm['root_name'],'native_root_train_export_signatures':True,'new_event':event.payload(),'source_and_input_bytes_unchanged':True,'no_model_or_storage_calls':True},indent=2,sort_keys=True)+'\n');print('PASS native A13 fresh namespace only, same science/source/ZIP')


def retirement_path_proof():
 from dataclasses import replace
 from unittest.mock import patch
 import experiments.post_training.identity_path_b.ablation_launch as launch
 from experiments.post_training.identity_path_b.ablation import frozen_names,AblationArm
 from experiments.post_training.identity_path_b.checkpoints import event_bytes
 plan=json.loads((P/'launch-plan.json').read_text());training=plan['arm']['training_kwargs'];event=CheckpointEvent(**plan['arm']['final_checkpoint']);root=training['run_root']
 source=json.loads(Path('/home/karan/Dropbox/marin-notes/snowball-identity/t310/rcd-lr1e-5-r1-receipts/source-receipt-af484f49.json').read_text())['tensors'];frozen=frozen_names(AblationArm('A13'))
 audit={'source_receipt_sha256':launch.SOURCE_RECEIPT['sha256'],'schema':'path_b_storage_v1','frozen_native_tensors':104,'trained_master_cast_tensors':398,'tensors':{n:{'frozen_source':n in frozen}for n in source}}
 objects={event.hf_export+'/path-b-storage-audit.json':json.dumps(audit).encode(),event.native_checkpoint+'/metadata.json':b'{"step":35,"is_temporary":true}'};present={event.native_checkpoint};operations=[]
 class MemoryPath:
  def __init__(self,uri):self.uri=str(uri)
  def __truediv__(self,other):return MemoryPath(self.uri+'/'+str(other))
  def read_bytes(self):return objects[self.uri]
  def rmtree(self):operations.append(('retire',self.uri));present.remove(self.uri)
  def exists(self):return self.uri in present or self.uri in objects
  def write_bytes(self,raw):objects[self.uri]=raw;operations.append(('write',self.uri))
 def checked(e):
  assert e==event
 with patch.object(launch,'StoragePath',MemoryPath),patch.object(launch,'checked_export_event',checked):
  launch.retire_native(event,training)
  assert operations==[('retire',event.native_checkpoint),('write',root+'/native-retired.json')]
  assert objects[root+'/native-retired.json']==event_bytes(event)
  good=list(operations);operations.clear()
  bad_root=root.replace('/bfcl-ablation-a13-','/a13-');bad_training={**training,'run_root':bad_root};bad_event=replace(event,native_checkpoint=event.native_checkpoint.replace(root,bad_root),hf_export=event.hf_export.replace(root,bad_root))
  for e,t in [(bad_event,bad_training),(replace(event,native_checkpoint=event.native_checkpoint.replace('step-35','step-34')),training)]:
   try:launch.retire_native(e,t)
   except ValueError:pass
   else:raise AssertionError('Retirement guard accepted an invalid path')
   assert operations==[]
 proof={'actual_retire_native_exercised':True,'valid_fresh_namespace':root,'only_exact_final_native_retired':good[0][1],'retired_marker_matches_event_bytes':True,'old_invalid_namespace_and_wrong_step_rejected_before_operations':True,'all_filesystem_export_proofs':'in-memory synthetic502audit/native-metadata fixtures; not actual readiness','actual_remote_storage_or_deletion_calls':0}
 (P/'retirement-path-guard-proof.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n');print('PASS actual retire_native synthetic guard positive and negatives')
if __name__=='__main__':
 main()
 retirement_path_proof()
