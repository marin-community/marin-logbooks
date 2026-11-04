"""Synthetic metadata-boundary tests only; never claim actual A9/A10 HF readiness."""
from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile
from types import SimpleNamespace
from experiments.post_training.identity_path_b.native_boundary_scope import owner
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
HERE=Path(__file__).resolve().parent;B=HERE.parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
 spec=importlib.util.spec_from_file_location('future_readiness_guard',HERE/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 plan=json.loads((HERE/'launch-plan.json').read_bytes());proof=[]
 real_here=m.HERE;real_submission=m.submission
 try:
  with tempfile.TemporaryDirectory(prefix='SYNTHETIC_NOT_ACTUAL_A9_A10_READINESS-') as directory:
   root=Path(directory);(root/'review-receipt.json').write_text(json.dumps({'files':{}}));(root/'launch-plan.json').write_text(json.dumps(plan));m.HERE=root
   for label in ['A9','A10']:
    item=owner.target(label);event=CheckpointEvent(**item['event']);d=root/label;d.mkdir();runroot=event.hf_export.rsplit('/hf/',1)[0]
    audit=(B/'ablation-a8-r1/A8a/actual-metadata/audit.json').read_bytes();index=(B/'ablation-a8-r1/A8a/actual-metadata/model.safetensors.index.json').read_bytes()
    eventraw=(json.dumps(event.payload())+'\n').encode();objects={
     'audit.json':(event.hf_export+'/path-b-storage-audit.json',audit),
     'event.json':(event.hf_export+'/path-b-checkpoint-event.json',eventraw),
     'model.safetensors.index.json':(event.hf_export+'/model.safetensors.index.json',index),
     'native-retired.json':(runroot+'/native-retired.json',eventraw),
     'bfcl-ready.json':(runroot+'/bfcl-ready.json',json.dumps({'event':event.payload()}).encode()),
     'export-terminal.json':(runroot+'/workflow/export-terminal.json',b'{"state":"succeeded"}')}
    metadata={}
    for name,(uri,raw) in objects.items():
     p=d/name;p.write_bytes(raw);metadata[name]=dict(uri=uri,path=str(p),bytes=len(raw),sha256=sha(raw))
    record=dict(arm=label,HF=event.hf_export,metadata=metadata,fixture_only='SYNTHETIC, NOT ACTUAL TRAINING/EXPORT/READINESS')
    receipt=d/'SYNTHETIC-readiness.json';receipt.write_text(json.dumps(record));digest=sha(receipt.read_bytes());expected=sha(audit)
    assert m.checked_readiness(label,receipt,digest)==expected
    request=m.request(label,{'CW_KEY_ID':'fixture-only','CW_KEY_SECRET':'fixture-only'},receipt,digest)
    command=request['entrypoint'].command[-1];assert expected in command and digest in command
    calls=[]
    m.submission=lambda **kwargs:calls.append(kwargs)
    for case in ['wrong_target','tampered_receipt','tampered_local_audit','wrong_final_event','missing_digest']:
     data=copy.deepcopy(record);localraw=audit
     if case=='wrong_final_event':
      bad=dict(event.payload(),completed_updates=1);raw=json.dumps(bad).encode();p=d/'wrong-event.json';p.write_bytes(raw);data['metadata']['event.json'].update(path=str(p),bytes=len(raw),sha256=sha(raw))
     raw=json.dumps(data).encode();p=d/('SYNTHETIC-'+case+'.json');p.write_bytes(raw);h=sha(raw)
     if case=='tampered_receipt':h='f'*64
     if case=='tampered_local_audit':(d/'audit.json').write_bytes(b'{}')
     if case=='missing_digest':h=None
     selected=('A10' if label=='A9' else 'A9') if case=='wrong_target' else label
     try:m.request(selected,{'CW_KEY_ID':'fixture-only','CW_KEY_SECRET':'fixture-only'},p,h)
     except (ValueError,KeyError,TypeError):pass
     else:raise AssertionError('Accepted invalid readiness '+case)
     assert not calls
     (d/'audit.json').write_bytes(localraw)
    m.submission=real_submission
    # Real CPU owner must reject wrong fullGET audit before requesting any GPU child.
    actual_storage=owner.StoragePath;actual_event_check=owner.checked_export_event;actual_context=owner.regional_task_context
    submissions=[]
    class LocalStorage:
     def __init__(self,uri):self.uri=uri
     def exists(self):return False
     def __truediv__(self,child):return LocalStorage(self.uri+'/'+child)
     def read_bytes(self):
      assert self.uri==event.hf_export+'/path-b-storage-audit.json';return audit
     def write_bytes(self,raw):raise AssertionError('Invalid audit must fail before output writes')
    owner.StoragePath=LocalStorage;owner.checked_export_event=lambda event:b'fixture-only';owner.regional_task_context=lambda:submissions.append('requested')
    try:
     try:owner.screen_worker(label,plan['targets'][label]['root_name'],plan['targets'][label]['output'],'a'*64,digest)
     except ValueError as error:assert 'audit' in str(error).lower()
     else:raise AssertionError('CPU accepted wrong audit')
     assert not submissions
    finally:owner.StoragePath=actual_storage;owner.checked_export_event=actual_event_check;owner.regional_task_context=actual_context
    proof.append(dict(target=label,valid_SYNTHETIC_metadata_and_exact_parent_binding='PASS',wrong_target_tampered_missing_negatives='PASS_ZERO_SUBMISSIONS',CPU_wrong_audit_rejected_before_GPU='PASS',actual_HF_readiness=False))
 finally:m.HERE=real_here;m.submission=real_submission
 out=dict(scope='SYNTHETIC metadata/type boundary only; oldA8a audit/index reused as localfixtures, NOT ACTUAL A9/A10 weights/audit/readiness',targets=proof,model_GPU_RNO_job_calls=0)
 (HERE/'synthetic-readiness-guard-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
