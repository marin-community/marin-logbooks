"""Construct the one admitted M51 native132 owner without submitting."""
from pathlib import Path
import hashlib,importlib.util,json
from experiments.post_training.identity_path_b.native_boundary_scope.owner import submission
H=Path(__file__).resolve().parent

def request(credentials):
 for name,item in json.loads((H/'review-receipt.json').read_bytes())['files'].items():
  raw=Path(name).read_bytes()
  if len(raw)!=item['bytes']or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('M51 native binding changed: '+name)
 spec=importlib.util.spec_from_file_location('readiness',H/'check_readiness.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.check()
 plan=json.loads((H/'launch-plan.json').read_bytes());t=plan['targets']['M51']
 return submission(label='M51',name=t['root_name'],output=t['output'],credentials=credentials,audit_sha256=t['merge_admission_sha256'],readiness_sha256=plan['actual_readiness']['sha256'])
