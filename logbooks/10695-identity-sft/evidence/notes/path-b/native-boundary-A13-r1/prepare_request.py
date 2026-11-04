from pathlib import Path
import hashlib,json
from experiments.post_training.identity_path_b.native_boundary_scope.owner import submission
H=Path(__file__).resolve().parent

def request(label,credentials):
 for name,item in json.loads((H/'review-receipt.json').read_bytes())['files'].items():
  raw=Path(name).read_bytes()
  if len(raw)!=item['bytes']or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Native binding changed: '+name)
 p=json.loads((H/'launch-plan.json').read_bytes());t=p['targets'][label];ref=p['actual_readiness'][label];raw=Path(ref['path']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Require exact own readiness')
 r=json.loads(raw)
 if r['HF']!=t['model']:raise ValueError('Wrong target readiness')
 for item in r['metadata'].values():
  raw=Path(item['path']).read_bytes()
  if len(raw)!=item['bytes']or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Actual HF metadata changed')
 digest=t.get('merge_admission_sha256',t.get('actual_audit_sha256'))
 return submission(label=label,name=t['root_name'],output=t['output'],credentials=credentials,audit_sha256=digest,readiness_sha256=ref['sha256'])
