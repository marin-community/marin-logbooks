from pathlib import Path
import ast,copy,hashlib,inspect,io,json,os,shlex,subprocess,sys,tempfile,zipfile
import requests
from iris.cluster.client.bundle import create_workspace_zip
from iris.client.client import IrisClient
from experiments.post_training.identity_path_b.identity_scope import run as original
from experiments.post_training.identity_path_b.native_boundary_scope import run,owner
from experiments.post_training.identity_t2.capture import checked_tokenizer
HERE=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 plan=json.loads((HERE/'launch-plan.json').read_text());root=Path(plan['workspace']);scope='experiments/post_training/identity_path_b/identity_scope/'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin']
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
 parent=plan['parent_pin'];archive=root/scope/'payload.zip';oldraw=subprocess.check_output(['git','show',parent+':'+scope+'payload.zip'],cwd=root)
 assert archive.read_bytes()==oldraw and sha(archive)==original.PAYLOAD_SHA256=='200bf8a90cc7d6ec6df5fb6571547be375cc26e6d97a8b6cbcbf4fb20915b2a1'
 with zipfile.ZipFile(io.BytesIO(oldraw)) as z:payload_members={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
 for name in [scope+'run.py','experiments/post_training/identity_path_b/native_boundary_scope/run.py','experiments/post_training/identity_path_b/native_boundary_scope/prefixes.json']:
  assert subprocess.check_output(['git','show',parent+':'+name],cwd=root)==(root/name).read_bytes()
 cache=Path('/mnt/storage/karan-scratch/margay-t2-client-tmp/levanter_tokenizers/s3:/marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a/hf-hub-1.21.0');tok=checked_tokenizer(cache)
 evidence=run.prefixes();proofs=[];creds={'CW_KEY_ID':'fixture-only','CW_KEY_SECRET':'fixture-only'}
 for label,item in plan['targets'].items():
  req=owner.submission(label=label,name=item['root_name'],output=item['output'],credentials=creds,audit_sha256=item['actual_audit_sha256'],readiness_sha256='b'*64);serve=owner.serve_submission(label,item['root_name'],creds,item['actual_audit_sha256'],'b'*64)
  for r in [req,serve]:
   inspect.signature(IrisClient.submit).bind(None,**r);assert r['max_retries_failure']==r['max_retries_preemption']==0 and not r['entrypoint'].workdir_files
  assert req['timeout'].to_seconds()==176400 and serve['timeout'].to_seconds()==43200
  assert req['priority_band']==serve['priority_band']==owner.job_pb2.PRIORITY_BAND_PRODUCTION
  cfg=owner.service_config(label,item['root_name']);assert cfg.model.weights==item['model'] and cfg.model.tokenizer is None and cfg.model.chat_template_content is None
  saved={}
  def send(session,request,**kwargs):
   body=json.loads(request.body)
   prefix=tok.apply_chat_template(body['messages'],tokenize=True,add_generation_prompt=True,continue_final_message=False,return_dict=False,**body['chat_template_kwargs'])
   response=requests.Response();response.status_code=200
   if request.url.endswith('/tokenize'):result=dict(tokens=prefix)
   else:
    ids=[128002,11,128003,12,128009] if body['chat_template_kwargs']['enable_thinking'] else [12,128009]
    result=dict(prompt_token_ids=prefix,choices=[dict(token_ids=ids,message=dict(role='assistant',content='  RAW\n',reasoning_content=''),finish_reason='stop')],usage=dict(completion_tokens=len(ids)))
   response._content=json.dumps(result).encode();return response
  real=requests.Session.send;requests.Session.send=send
  try:
   with original.payload() as (r,h):
    tasks=original.jobs(r,h);records=original.generate('http://fixture',item['model'],r,h,lambda i,x:saved.update({i:x}),session_factory=run.NativeSession)
  finally:requests.Session.send=real
  assert len(records)==len(saved)==132
  for record,(row,thinking) in zip(records,tasks,strict=True):
   assert record['row_id']==row.id and record['thinking']==thinking and record['original_messages']==row.messages and record['coverage']=='response_received'
   token,completion=record['raw_http'];body=completion['request'];old=completion['original_request']
   assert body=={**old,'skip_special_tokens':False,'return_token_ids':True}
   assert old==dict(model=item['model'],messages=row.messages,temperature=0.0,max_tokens=4096 if thinking else 1024 if row.set=='identity' else 2048,chat_template_kwargs=dict(enable_thinking=thinking))
   ids=tok.apply_chat_template(row.messages,tokenize=True,add_generation_prompt=True,return_dict=False,enable_thinking=thinking)
   assert ids==evidence[run.prompt_key(old)]['original_ids']==token['native_CPU_prefix_ids']==json.loads(token['response']['body'])['tokens']==completion['returned_prompt_ids']
   assert completion['current_assistant_prefix']==run.assistant_prefix(ids)
   if thinking:assert completion['current_assistant_prefix']['prompt_owned_start_count']==0
   raw=json.loads(completion['response']['body']);assert completion['provider_message_unmodified']==raw['choices'][0]['message']
   assert completion['generated_ids']==raw['choices'][0]['token_ids']
   assert completion['native_boundary']==run.boundary(raw['choices'][0]['token_ids'])
  proofs.append(dict(label=label,model=item['model'],records=132,CPU_server_returned_prefixes='MOCK_HTTP_ALL132_PASS',request_delta='ONLY_TWO_FLAGS',original_contexts_modes_limits='PASS',native_owner_serve='PASS',parent_command=req['entrypoint'].command,child_command=serve['entrypoint'].command,audit_sha256=item['actual_audit_sha256'],readiness_sha256='b'*64,no_small_workdir='PASS',no_template_override_or_prefill=True))
 bundle=create_workspace_zip(root)
 with zipfile.ZipFile(io.BytesIO(bundle)) as z,tempfile.TemporaryDirectory(prefix='native-boundary-actual-zip-') as directory:
  dest=Path(directory);z.extractall(dest)
  members=[scope+'payload.zip',scope+'run.py']+['experiments/post_training/identity_path_b/native_boundary_scope/'+n for n in ['owner.py','run.py','targets.json','prefixes.json']]
  for name in members:assert z.read(name)==(root/name).read_bytes()
  env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONSAFEPATH='1',IRIS_PYTHON=sys.executable);env['PYTHONPATH']=':'.join([str(dest),*[str(p) for p in (dest/'lib').glob('*/src')]])
  for label,item in plan['targets'].items():
   r=owner.submission(label=label,name=item['root_name'],output=item['output'],credentials=creds,audit_sha256=item['actual_audit_sha256'],readiness_sha256='b'*64);command=r['entrypoint'].command.copy();command[-1]+=' --inspect-only'
   result=subprocess.run(command,cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90)
   assert json.loads(result.stdout.strip().splitlines()[-1])==dict(target=label,model=item['model'],records=132,calls=0)
   child=owner.serve_submission(label,item['root_name'],creds,item['actual_audit_sha256'],'b'*64)
   words=shlex.split(child['entrypoint'].command[-1].splitlines()[-1]);assert words[:3]==['exec','$root_python','-u']
   cmd=[sys.executable,*words[2:],'--inspect-only']
   result=subprocess.run(cmd,cwd=dest,env=env,capture_output=True,text=True,check=True,timeout=90)
   assert json.loads(result.stdout.strip().splitlines()[-1])==dict(target=label,model=item['model'],records=132,calls=0)
   for flag in ['--audit-sha256','--readiness-sha256']:
    bad=list(cmd);at=bad.index(flag);del bad[at:at+2]
    result=subprocess.run(bad,cwd=dest,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==2 and flag in result.stderr
  count=len(z.namelist())
 out=dict(code_pin=plan['code_pin'],native_bundle_sha256=hashlib.sha256(bundle).hexdigest(),native_bundle_bytes=len(bundle),members=count,original_payload_members_byte_equal=payload_members,parentNativeSession_prefix_payload_generate_bytes_exact=True,targets=proofs,actual_extracted_parent_and_child_CLI_all1='PASS',live_prefix_or_model_proof=False,model_GPU_remote_calls=0)
 (HERE/'native-one-binding-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['original_payload_members_byte_equal','targets']}))
if __name__=='__main__':main()
