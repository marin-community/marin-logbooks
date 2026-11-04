"""Join actual A8 metadata to the sealed plan; no tensor/model or remote reads."""
import argparse,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['A8a','A8b'],required=True);p.add_argument('--ready-sha256',required=True);args=p.parse_args()
HERE=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
ready=HERE/args.arm/'actual-audited-HF-ready-margay-r1.json';assert sha(ready)==args.ready_sha256;r=json.loads(ready.read_text())
assert sha(HERE/'launch-plan.json')=='f527fb76e294211e33e5e28eb1183bc6788ae5414f229542e047c410332e2a65'
plan=json.loads((HERE/'launch-plan.json').read_text());target=next(a for a in plan['arms'] if a['arm']==args.arm)
for name,x in r['metadata'].items():
 f=Path(x['path']);assert f.stat().st_size==x['bytes'] and sha(f)==x['sha256'],name
metadata={name:json.loads(Path(x['path']).read_text()) for name,x in r['metadata'].items()}
t=metadata['training-result.json'];kw=target['training_kwargs'];assert t['arm']==args.arm and t['training_pin']==kw['training_pin'] and t['completed_updates']==36
assert t['packed_sha256']==kw['packed_sha256'] and t['exposure_sha256']==kw['exposure_sha256'] and t['special_rows_frozen'] is True
assert [x['completed_updates'] for x in t['updates']]==list(range(1,37));assert all(math.isfinite(x['loss']) for x in t['updates'])
base=json.loads((HERE.parent/'ablation-a6-r1/actual-training-result.json').read_text());assert t['frozen_special_row_sha256']==base['frozen_special_row_sha256'];assert t['fully_frozen_hf_names']==base['fully_frozen_hf_names'] and len(set(t['fully_frozen_hf_names']))==104
expected=CheckpointEvent(**target['final_checkpoint']).payload()
assert metadata['final-checkpoint-event.json']==metadata['event.json']==metadata['native-retired.json']==metadata['bfcl-ready.json']['event']==expected
assert metadata['export-terminal.json']['state']=='succeeded' and metadata['export-terminal.json']['job'].endswith('/export-36')
a=metadata['audit.json'];assert a['schema']=='path_b_storage_v1' and a['source_receipt_sha256']==plan['source_receipt']['sha256']
assert a['frozen_native_tensors']==104 and a['trained_master_cast_tensors']==398 and len(a['tensors'])==502
frozen={name for name,x in a['tensors'].items() if x['frozen_source']};assert frozen==set(t['fully_frozen_hf_names'])
assert all(len(x['export_sha256'])==64 for x in a['tensors'].values())
assert all('master_sha256' in x for x in a['tensors'].values() if not x['frozen_source'])
index=metadata['model.safetensors.index.json'];names=set(index['weight_map'].values());assert len(names)==39 and len(index['weight_map'])==502
present=r['all_indexed_shards_present'];assert {x['name'] for x in present}==names and all(x['bytes']>0 for x in present)
report=dict(utc=datetime.now(timezone.utc).isoformat(),arm=args.arm,status='PASS_ACTUAL_TRAINING_EXPORT_LOCAL_METADATA_JOINS',ready_receipt_sha256=sha(ready),verified_local_metadata=len(metadata),completed_updates=36,ordered_finite_loss_records=36,packed_sha256=t['packed_sha256'],exposure_sha256=t['exposure_sha256'],training_pin=t['training_pin'],frozen_full_names=104,special_row_hashes_exact_A6=True,exact_native_HF_retired_ready_events=True,audit_frozen_native=104,audit_master_casts=398,audit_tensors=502,HF_index_shards=39,actual_HF=expected['hf_export'],exposure=expected['exposure'],limitations='Independent LOCAL metadata/receipt/hash joins only. Actual tensor-value audits and shard HEAD existence are worker/operator evidence; no new tensor read, network, native lease/delete, model call, quality claim or retry authority.',quality_or_release_PASS=False)
out=HERE/args.arm/'actual-metadata-independent-joins-genet-r1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print('SHA256',sha(out))
