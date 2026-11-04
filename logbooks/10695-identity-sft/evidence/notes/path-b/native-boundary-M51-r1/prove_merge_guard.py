from pathlib import Path
from unittest.mock import patch
import hashlib,importlib.util,json
from experiments.post_training.identity_path_b.native_boundary_scope import owner
H=Path(__file__).resolve().parent

def main():
 plan=json.loads((H/'launch-plan.json').read_text());target=plan['targets']['M51'];spec=importlib.util.spec_from_file_location('check',H/'check_readiness.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);ready=m.check();objects={item['uri']:Path(item['path']).read_bytes()for item in ready['metadata'].values()};reads=[]
 class Memory:
  def __init__(self,uri):self.uri=str(uri)
  def __truediv__(self,part):return Memory(self.uri+'/'+str(part))
  def read_bytes(self):reads.append(self.uri);return objects[self.uri]
 with patch.object(owner,'StoragePath',Memory):
  identity=owner.checked_identity('M51',target['root_name'],target['merge_admission_sha256']);assert identity['kind']=='linear-cpu-merge'and identity['training_event']is None and identity['master_cast_audit']is None
  passed=list(reads);reads.clear()
  for digest in ['a'*64,'','A'*64]:
   try:owner.checked_identity('M51',target['root_name'],digest)
   except ValueError:pass
   else:raise AssertionError('Wrong/missing admission digest accepted')
   assert not reads
  for name in ['merge-admission.json','merge-manifest.json','tokenizer_config.json']:
   uri=ready['metadata'][name]['uri'];original=objects[uri];objects[uri]=original+b' '
   try:owner.checked_identity('M51',target['root_name'],target['merge_admission_sha256'])
   except ValueError:pass
   else:raise AssertionError('Changed merge metadata accepted')
   objects[uri]=original;reads.clear()
 out=dict(result='PASS',actual_readiness_sha256=plan['actual_readiness']['sha256'],actual_admission_manifest_and_5_HF_metadata_checked=passed,wrong_missing_digest_beforeIO=True,tampered_report_manifest_tokenizer_reject=True,honest_merge_no_training_event_or_mastercast=True,model_storage_remote_calls=0)
 (H/'actual-merge-guard-proof.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS M51 actualmerge guard + beforeIO negatives; no remote calls')
if __name__=='__main__':main()
