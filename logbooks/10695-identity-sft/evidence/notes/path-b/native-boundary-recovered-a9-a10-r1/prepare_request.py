"""Bind future actual target readiness without transporting a small candidate file."""
from pathlib import Path
import hashlib,json
from experiments.post_training.identity_path_b.native_boundary_scope.owner import submission,target,checked_audit_sha256
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
HERE=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked_readiness(label,receipt_path,receipt_sha256):
    checked_audit_sha256(receipt_sha256)
    raw=Path(receipt_path).read_bytes()
    if sha(raw)!=receipt_sha256:raise ValueError('Actual immutable readiness receipt hash mismatch')
    record=json.loads(raw);item=target(label);event=CheckpointEvent(**item['event'])
    if record.get('arm')!=label or record.get('HF')!=item['model']:
        raise ValueError('Readiness receipt belongs to a different target/HF')
    admitted={}
    for key,filename in [('audit.json','path-b-storage-audit.json'),('event.json','path-b-checkpoint-event.json')]:
        metadata=record['metadata'][key]
        if metadata['uri']!=item['model']+'/'+filename:raise ValueError('Readiness metadata points to a different target HF')
        data=Path(metadata['path']).read_bytes()
        if len(data)!=metadata['bytes'] or sha(data)!=metadata['sha256']:raise ValueError('Readiness local FULLGET bytes changed')
        admitted[key]=json.loads(data)
    if admitted['event.json']!=event.payload():raise ValueError('Actual final event differs from frozen target plan')
    audit=admitted['audit.json']
    if audit['schema']!='path_b_storage_v1' or audit['source_receipt_sha256']!='af484f49ac2c1d67989495475bcc79bf7519d09281c8087321d906b67139fa97':raise ValueError('Audit/source provenance mismatch')
    if len(audit['tensors'])!=502 or audit['frozen_native_tensors']!=104 or audit['trained_master_cast_tensors']!=398:raise ValueError('Incomplete target audit')
    if sum(bool(x['frozen_source']) for x in audit['tensors'].values())!=104:raise ValueError('Target frozen classification mismatch')
    for tensor in audit['tensors'].values():checked_audit_sha256(tensor['export_sha256'])
    expected_root=event.hf_export.rsplit('/hf/',1)[0]
    metadata=record['metadata']
    required={'model.safetensors.index.json':event.hf_export+'/model.safetensors.index.json',
              'native-retired.json':expected_root+'/native-retired.json',
              'bfcl-ready.json':expected_root+'/bfcl-ready.json',
              'export-terminal.json':expected_root+'/workflow/export-terminal.json'}
    for key,uri in required.items():
        item=metadata[key];data=Path(item['path']).read_bytes()
        if item['uri']!=uri or len(data)!=item['bytes'] or sha(data)!=item['sha256']:
            raise ValueError('Readiness index/marker target or bytes mismatch')
        admitted[key]=json.loads(data)
    index=admitted['model.safetensors.index.json']['weight_map']
    if set(index)!=set(audit['tensors']) or len(set(index.values()))!=39:
        raise ValueError('Audited target index must cover all502 tensors and39 shards')
    if admitted['native-retired.json']!=event.payload() or admitted['bfcl-ready.json']['event']!=event.payload():
        raise ValueError('Retirement/ready markers belong to a different final target')
    if admitted['export-terminal.json']['state']!='succeeded':raise ValueError('Final export is not terminal successful')
    return record['metadata']['audit.json']['sha256']

def request(label,credentials,readiness_receipt,readiness_sha256):
    for name,item in json.loads((HERE/'review-receipt.json').read_text())['files'].items():
        raw=Path(name).read_bytes()
        if len(raw)!=item['bytes'] or sha(raw)!=item['sha256']:raise ValueError('Immutable native future packet mismatch: '+name)
    digest=checked_readiness(label,readiness_receipt,readiness_sha256)
    item=json.loads((HERE/'launch-plan.json').read_text())['targets'][label]
    return submission(label=label,name=item['root_name'],output=item['output'],credentials=credentials,
                      audit_sha256=digest,readiness_sha256=readiness_sha256)
