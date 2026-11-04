from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent

def check():
 plan=json.loads((H/'launch-plan.json').read_bytes());entry=plan['actual_readiness'];raw=Path(entry['path']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Actual recovered A12 readiness changed')
 ready=json.loads(raw);target=plan['targets']['A12']
 if ready['HF']!=target['model']or ready['arm']!='A12'or ready['actual_RNO_bucket_landing_verified_NOT_just_declared_URI']is not True or ready['original_exact_native17_absent_after_worker_retirement']!=target['event']['native_checkpoint']:raise ValueError('Require this exact own recovered18 readiness')
 for name,item in ready['metadata'].items():
  value=Path(item['path']).read_bytes()
  if len(value)!=item['bytes']or hashlib.sha256(value).hexdigest()!=item['sha256']:raise ValueError('Admitted HF metadata changed: '+name)
 if ready['metadata']['audit.json']['sha256']!=target['actual_audit_sha256']:raise ValueError('Audit binding differs')
 event=json.loads(Path(ready['metadata']['event.json']['path']).read_bytes())
 if any(event[k]!=v for k,v in target['event'].items()):raise ValueError('Wrong original18 recovered event')
 return ready
