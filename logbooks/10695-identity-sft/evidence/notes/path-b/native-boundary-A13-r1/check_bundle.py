from pathlib import Path
from unittest.mock import patch
import ast,hashlib,inspect,io,json,os,re,shlex,subprocess,sys,tempfile,zipfile
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from experiments.post_training.identity_path_b.native_boundary_scope import owner
from experiments.post_training.identity_path_b import async_launch
H=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((H/'launch-plan.json').read_text());W=Path(p['workspace']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==p['code_pin'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
 scope='experiments/post_training/identity_path_b/';unchanged=[scope+'identity_scope/payload.zip',scope+'identity_scope/run.py',scope+'native_boundary_scope/run.py',scope+'native_boundary_scope/prefixes.json']
 for rel in unchanged:assert subprocess.check_output(['git','show',p['parent_pin']+':'+rel],cwd=W)==(W/rel).read_bytes()
 rel=scope+'native_boundary_scope/owner.py';old=subprocess.check_output(['git','show',p['parent_pin']+':'+rel],cwd=W,text=True);new=(W/rel).read_text();merge=next(iter(p['targets'])).startswith('M')
 def normalize(s):
  s=re.sub(r'TARGETS_SHA256 = "[a-f0-9]+"','TARGETS_SHA256 = "PIN"',s);s=re.sub(r'choices=\([^\n]*\), required=True','choices=("TARGETS",), required=True',s)
  if merge:s=s.replace('[0.51]','[item["coefficient"]]').replace('exact finite/restored502 M51 merge report','exact finite/restored502 target merge report')
  return ast.dump(ast.parse(s),include_attributes=False)
 assert normalize(old)==normalize(new)
 proofs=[];creds=dict(CW_KEY_ID='fixture',CW_KEY_SECRET='fixture')
 for label,t in p['targets'].items():
  ref=p['actual_readiness'][label];assert sha(Path(ref['path']))==ref['sha256'];r=json.loads(Path(ref['path']).read_text());assert r['HF']==t['model'];objects={}
  for item in r['metadata'].values():
   raw=Path(item['path']).read_bytes();assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256'];objects[item['uri']]=raw
  reads=[]
  class Memory:
   def __init__(self,uri):self.uri=str(uri)
   def __truediv__(self,k):return Memory(self.uri+'/'+str(k))
   def read_bytes(self):reads.append(self.uri);return objects[self.uri]
  digest=t.get('merge_admission_sha256',t.get('actual_audit_sha256'))
  with patch.object(owner,'StoragePath',Memory),patch.object(async_launch,'StoragePath',Memory):
   ident=owner.checked_identity(label,t['root_name'],digest);passed=list(reads)
   if merge:assert ident['kind']=='linear-cpu-merge'and ident['training_event']is None and ident['master_cast_audit']is None
   else:assert ident['event']['completed_updates']==36 and ident['event']['hf_export']==t['model']
   for key in (['merge-admission.json','merge-manifest.json','tokenizer_config.json']if merge else['event.json','audit.json','tokenizer_config.json']):
    uri=r['metadata'][key]['uri'];raw=objects[uri];objects[uri]=raw+b' '
    if key=='event.json':wrong=json.loads(raw);wrong['completed_updates']=72;objects[uri]=json.dumps(wrong).encode()
    try:owner.checked_identity(label,t['root_name'],digest)
    except ValueError:pass
    else:raise AssertionError('Tampered target provenance accepted')
    objects[uri]=raw
   for invalid in ['', 'X'*64]:
    try:owner.checked_audit_sha256(invalid)
    except ValueError:pass
    else:raise AssertionError('Malformed digest accepted')
  parent=owner.submission(label=label,name=t['root_name'],output=t['output'],credentials=creds,audit_sha256=digest,readiness_sha256=ref['sha256']);child=owner.serve_submission(label,t['root_name'],creds,digest,ref['sha256'])
  for req in [parent,child]:
   inspect.signature(IrisClient.submit).bind(None,**req);assert req['priority_band']==owner.job_pb2.PRIORITY_BAND_PRODUCTION and req['max_retries_failure']==req['max_retries_preemption']==0 and not req['entrypoint'].workdir_files
  assert parent['timeout'].to_seconds()==176400 and child['timeout'].to_seconds()==43200;cfg=owner.service_config(label,t['root_name']);assert cfg.model.weights==t['model']and cfg.model.tokenizer is None and cfg.model.chat_template_content is None and cfg.iris.priority==owner.job_pb2.PRIORITY_BAND_PRODUCTION
  proofs.append(dict(label=label,actual_readiness_sha256=ref['sha256'],actual_digest=digest,checked_actual_guard_uris=passed,parent_command=parent['entrypoint'].command,child_command=child['entrypoint'].command,priority='production'))
 bundle=create_workspace_zip(W)
 with tempfile.TemporaryDirectory(prefix='native-target-only-')as td,zipfile.ZipFile(io.BytesIO(bundle))as z:
  dest=Path(td);z.extractall(dest);env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONSAFEPATH='1',IRIS_PYTHON=sys.executable);env['PYTHONPATH']=':'.join([str(dest),*[str(q)for q in(dest/'lib').glob('*/src')]])
  for item in proofs:
   for role in ['parent','child']:
    command=item[role+'_command'];words=shlex.split(command[-1].splitlines()[-1]);assert words[0]=='exec';cmd=[sys.executable,*words[2:],'--inspect-only'];ran=subprocess.run(cmd,cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90);assert json.loads(ran.stdout.strip().splitlines()[-1])==dict(target=item['label'],model=p['targets'][item['label']]['model'],records=132,calls=0)
 out=dict(result='PASS',code_pin=p['code_pin'],native_bundle_sha256=hashlib.sha256(bundle).hexdigest(),native_bundle_bytes=len(bundle),unchanged_original_payload_native_session_prefix_generation_bytes=True,target_only_owner_AST_delta=True,inherited_all132_CPU_mockHTTP_proof=p['parent_review'],all_targets_actual_guards_and_negatives_production_constructors_and_extracted_parent_child_CLIs=proofs,remote_calls=0,live_prefix_or_worker_runtime_attestation=False)
 (H/'native-target-binding-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='all_targets_actual_guards_and_negatives_production_constructors_and_extracted_parent_child_CLIs'}))
if __name__=='__main__':main()
