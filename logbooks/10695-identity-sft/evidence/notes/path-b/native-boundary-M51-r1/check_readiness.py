from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent

def check():
 plan=json.loads((H/'launch-plan.json').read_bytes());entry=plan['actual_readiness'];raw=Path(entry['path']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Actual M51 readiness receipt changed')
 ready=json.loads(raw);target=plan['targets']['M51']
 if ready['HF']!=target['model']or ready['label']!='M51'or ready['provenance_kind']!='linear-cpu-merge'or ready['fake_training_event_or_master_cast_audit']is not False or ready['ready_for_authorized_PRODUCTION_BFCL5_native132_first_original_counts_candidate_only_maths']is not True:raise ValueError('Require this exact actual M51 merge readiness')
 for name,item in ready['metadata'].items():
  value=Path(item['path']).read_bytes()
  if len(value)!=item['bytes']or hashlib.sha256(value).hexdigest()!=item['sha256']:raise ValueError('Actual merge metadata changed: '+name)
 if ready['metadata']['merge-admission.json']['sha256']!=target['merge_admission_sha256']or ready['metadata']['merge-manifest.json']['sha256']!=target['merge_manifest_sha256']:raise ValueError('Merge readiness/report/manifest binding differs')
 return ready
