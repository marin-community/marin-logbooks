from pathlib import Path
from concurrent.futures import Future
import hashlib,json,tempfile,threading
import pytest
from iris.testing.journeys.world import journey_world
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from experiments.post_training.identity_path_b import write_recovery as recovery,worker,async_launch,ablation_launch,export_preflight
from experiments.post_training.identity_path_b.ablation import AblationArm,export_worker,frozen_names
from experiments.post_training.identity_path_b.checkpoints import event_bytes
from experiments.post_training.identity_path_b.testing_support import ExecutionBoundary
B=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');HERE=Path(__file__).resolve().parent

def case(mode):
 with tempfile.TemporaryDirectory(prefix='pathb-recovery-lifecycle-') as td,pytest.MonkeyPatch.context() as mp:
  tmp=Path(td);item=recovery.binding('A9');original=recovery.checkpoint_event(item['original_event']);event=recovery.checkpoint_event(item['event']);training=item['original_training']
  def local(uri):
   p=tmp/'storage'/str(uri).removeprefix('s3://');p.parent.mkdir(parents=True,exist_ok=True);return StoragePath(str(p))
  def recording_path(uri):
   if mode=='submission_record_failure' and str(uri).endswith('/workflow/export-submitted.json'):return Denied(local(uri))
   return local(uri)
  for mod in [recovery,worker,ablation_launch]:mp.setattr(mod,'StoragePath',local)
  mp.setattr(async_launch,'StoragePath',recording_path)
  class Denied:
   def __init__(self,path):self.path=path
   def __truediv__(self,name):return Denied(self.path/name)
   def exists(self):return self.path.exists()
   def write_bytes(self,raw):raise PermissionError('Storage write suspended fixture')
  def probe_path(uri):
   p=local(uri)
   return Denied(p) if mode=='probe_failure' else p
  mp.setattr(export_preflight,'StoragePath',probe_path)
  stage=json.loads((B/'ablation-a9-r1/stage-files.json').read_text())['files'];exposure=next(x for x in stage if x['uri']==training['exposure_path']);raw=Path(exposure['local']).read_bytes();assert hashlib.sha256(raw).hexdigest()==training['exposure_sha256'];local(training['exposure_path']).write_bytes(raw)
  idx=json.loads((B/'ablation-a9-r1/original-native71-recovery-evidence-margay-r1/actual-three-metadata-fullget-index.json').read_text())
  evidence=item['original_evidence']
  for entry in idx['metadata']:
   raw=Path(entry['path']).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'];local(entry['uri']).write_bytes(raw)
  old_partial=local(original.hf_export+'/model-partial.safetensors');old_partial.write_bytes(b'PRESERVE FAILED ORIGINAL EXPORT')
  audit=dict(schema='path_b_storage_v1',source_receipt_sha256=training['source_receipt']['sha256'],frozen_native_tensors=104,trained_master_cast_tensors=398,tensors={**{n:dict(frozen_source=True) for n in frozen_names(AblationArm.GLM_ON_TENTH_72)},**{f'trained.{i}':dict(frozen_source=False) for i in range(398)}})
  if mode=='audit_failure':audit['trained_master_cast_tensors']=397
  future=Future()
  with journey_world(tmp/'controller',mp) as world:
   parent=world.submit('export-only-recovery-owner');world.settle();execution=ExecutionBoundary(world,parent);mp.setattr(recovery,'native_submit',execution.submit)
   def run():
    try:recovery.export_only_worker(runtime_bytes=b'{"xla_precision":"default"}');future.set_result(None)
    except BaseException as error:future.set_exception(error)
   thread=threading.Thread(target=run,daemon=True);thread.start()
   if mode=='probe_failure':
    with pytest.raises(PermissionError):future.result(timeout=15)
    assert not execution.jobs
   else:
    export=execution.get('export-72');assert set(execution.jobs)=={'export-72'} and export.kwargs['event']==event
    assert export.submit_options['lifetime']==7200 and world.job(parent).state==job_pb2.JOB_STATE_RUNNING and not future.done()
    assert local(original.native_checkpoint).exists() and old_partial.read_bytes()==b'PRESERVE FAILED ORIGINAL EXPORT'
    assert local(item['root']+'/workflow/export-preflight.json').exists()
    local(event.hf_export+'/path-b-checkpoint-event.json').write_bytes(event_bytes(event));local(event.hf_export+'/path-b-storage-audit.json').write_bytes(json.dumps(audit).encode())
    execution.finish('export-72',success=mode!='export_failure')
    if mode=='export_failure':
     with pytest.raises(RuntimeError):future.result(timeout=15)
    elif mode=='audit_failure':
     with pytest.raises(ValueError):future.result(timeout=15)
    else:future.result(timeout=15)
   thread.join(timeout=15);assert not thread.is_alive()
   assert old_partial.read_bytes()==b'PRESERVE FAILED ORIGINAL EXPORT'
   assert local(training['run_root']+'/final-checkpoint-event.json').read_bytes()==event_bytes(original)
   assert local(original.native_checkpoint).exists()==(mode not in ['success','submission_record_failure'])
   assert local(item['root']+'/bfcl-ready.json').exists()==(mode in ['success','submission_record_failure'])
   if mode in ['success','submission_record_failure']:
    assert json.loads(local(item['root']+'/native-retired.json').read_bytes())==event.payload()
    assert json.loads(local(item['root']+'/bfcl-ready.json').read_bytes())['event']==event.payload()
  return dict(case=mode,result='PASS',GPU_children=0 if mode=='probe_failure' else 1,training_children=0,native_retired_only_after_successful_full_audit=mode in ['success','submission_record_failure'],old_partial_HF_and_original_event_unchanged=True)

def main():
 outcomes=[case(mode) for mode in ['probe_failure','export_failure','audit_failure','submission_record_failure','success']]
 result=dict(result='PASS',cases=outcomes,boundary='Real Iris controller + native callable deserialization; local fake storage/GPU work only. Actual original72-update pinned metadata; synthetic new-export audit metadata, no actual recovery readiness claimed.',remote_model_storage_job_calls=0)
 (HERE/'local-recovery-lifecycle-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
