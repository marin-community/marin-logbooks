from pathlib import Path
import hashlib,inspect,io,json,os,subprocess,tempfile,zipfile
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from iris.rpc import job_pb2
from experiments.weight_merging.path_b_sweep import worker
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/karan/oa/marin-path-b-merge-sweep')
sha=lambda x:hashlib.sha256(x).hexdigest()
def main():
 pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 source=json.loads(worker.artifact('v01-source-receipt.json'))['tensors'];items=[]
 for label,alpha in zip(worker.LABELS,[.25,.49,.51,.75],strict=True):
  recipe=json.loads(worker.artifact(label+'.json'));assert recipe['parameters']==dict(method='average',coefficients=[alpha],density=1,scale=1,seed=0)
  assert recipe['tensor_coefficients']==recipe['row_overrides']=={}
  assert set(recipe['preserve_rows'])==worker.FROZEN|set(worker.SPECIAL)
  for n in worker.FROZEN:assert recipe['preserve_rows'][n]==list(range(source[n]['shape'][0]))
  for n in worker.SPECIAL:assert recipe['preserve_rows'][n]==list(range(128000,128256))
  request=worker.submission(label,pin,dict(CW_KEY_ID='fixture-only',CW_KEY_SECRET='fixture-only'))
  inspect.signature(IrisClient.submit).bind(None,**request)
  fn,args,kwargs=request['entrypoint'].resolve();assert fn is worker.run and kwargs==dict(label=label,code_revision=pin);inspect.signature(fn).bind(*args,**kwargs)
  assert request['priority_band']==job_pb2.PRIORITY_BAND_PRODUCTION
  assert request['timeout'].to_seconds()==43200 and request['max_retries_failure']==request['max_retries_preemption']==0
  assert request['constraints'] and request['entrypoint'].workdir_files
  items.append(dict(label=label,name=request['name'],output=recipe['output'],alpha=alpha,full104_restoration=True,router_biases_fp32=26,resources=str(request['resources']),constraints=str(request['constraints']),priority='production',native_signature=True))
 bundle=create_workspace_zip(ROOT)
 with zipfile.ZipFile(io.BytesIO(bundle)) as z:
  files=[p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/weight_merging/path_b_sweep').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
  for p in files:assert z.read(p)==(ROOT/p).read_bytes()
  for p in ['experiments/weight_merging/merge.py','lib/marin/src/marin/merging/checkpoint.py','lib/marin/src/marin/merging/arithmetic.py']:assert z.read(p)==(ROOT/p).read_bytes()
  with tempfile.TemporaryDirectory(prefix='genet-merge-native-') as d:
   extracted=Path(d);z.extractall(extracted)
   env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(extracted),*[str(p) for p in (extracted/'lib').glob('*/src')]])
   for label in worker.LABELS:
    raw=subprocess.check_output([os.sys.executable,'-m','experiments.weight_merging.path_b_sweep.worker','--label',label,'--code-revision',pin,'--inspect-only'],cwd=extracted,env=env)
    assert json.loads(raw)==json.loads(worker.artifact(label+'.json'))
  members=len(z.namelist())
 out=dict(code_pin=pin,native_bundle_sha256=sha(bundle),native_bundle_bytes=len(bundle),native_members=members,all_committed_package_members_exact=True,all_four_extracted_cli=True,roots=items,live_calls=0)
 (HERE/'native-four-roots-and-ZIP-proof.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
