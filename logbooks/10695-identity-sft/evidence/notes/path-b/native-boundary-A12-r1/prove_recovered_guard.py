from pathlib import Path
from unittest.mock import patch
import importlib.util,json
from experiments.post_training.identity_path_b.native_boundary_scope import owner
from experiments.post_training.identity_path_b import async_launch
H=Path(__file__).resolve().parent

def main():
 plan=json.loads((H/'launch-plan.json').read_text());target=plan['targets']['A12'];spec=importlib.util.spec_from_file_location('check',H/'check_readiness.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);ready=m.check();objects={e['uri']:Path(e['path']).read_bytes()for e in ready['metadata'].values()};reads=[]
 class Memory:
  def __init__(self,uri):self.uri=str(uri)
  def __truediv__(self,name):return Memory(self.uri+'/'+str(name))
  def read_bytes(self):reads.append(self.uri);return objects[self.uri]
 with patch.object(owner,'StoragePath',Memory),patch.object(async_launch,'StoragePath',Memory):
  identity=owner.checked_identity('A12',target['root_name'],target['actual_audit_sha256']);assert identity['event']['completed_updates']==18 and identity['event']['hf_export']==target['model'] and identity['event']['training_pin']=='69ddc88fb2ff4a6e45ee5cbbec866d9f86c2d068';passed=list(reads)
  for digest in ['', 'A'*64,'bad']:
   try:owner.checked_audit_sha256(digest)
   except ValueError:pass
   else:raise AssertionError('Missing or malformed digest accepted')
  for key in ['event.json','audit.json','tokenizer_config.json']:
   uri=ready['metadata'][key]['uri'];raw=objects[uri]
   if key=='event.json':obj=json.loads(raw);obj['completed_updates']=72;objects[uri]=json.dumps(obj).encode()
   else:objects[uri]=raw+b' '
   try:owner.checked_identity('A12',target['root_name'],target['actual_audit_sha256'])
   except ValueError:pass
   else:raise AssertionError('Tampered HF provenance accepted')
   objects[uri]=raw
 out=dict(result='PASS',actual_readiness_sha256=plan['actual_readiness']['sha256'],actual502_audit_event18_tokenizer_guard_read_uris=passed,malformed_digest_reject=True,wrong_horizon_event_tampered_audit_tokenizer_reject=True,no_new_training_replicate=True,remote_calls=0)
 (H/'actual-recovered-guard-proof.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS recovered original18 native guard +metadata negatives')
if __name__=='__main__':main()
