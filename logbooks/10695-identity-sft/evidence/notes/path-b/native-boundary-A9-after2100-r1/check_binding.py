from pathlib import Path
import ast,hashlib,importlib.util,inspect,io,json,zipfile
from datetime import datetime,timezone,timedelta
from iris.client.client import IrisClient
from iris.cluster.client.bundle import create_workspace_zip
from experiments.post_training.identity_path_b.native_boundary_scope import owner
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 plan=json.loads((HERE/'launch-plan.json').read_bytes());parent=Path(plan['previous_R2_receipt']['path']).parent
 old=ast.parse((parent/'prepare_request.py').read_text());new=ast.parse((HERE/'prepare_request.py').read_text())
 funcs=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 a,b=funcs(old),funcs(new);assert ast.dump(a['checked_readiness'])==ast.dump(b['checked_readiness'])
 request=b['request'];assert ast.dump(request.body.pop(0))==ast.dump(ast.parse('check_submit_clock()').body[0]);assert ast.dump(a['request'])==ast.dump(request)
 for name,item in json.loads((HERE/'review-receipt.json').read_bytes())['files'].items():
  p=Path(name);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 spec=importlib.util.spec_from_file_location('a9_quiet_native',HERE/'prepare_request.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 cutoff=datetime.fromisoformat(plan['not_before_utc']);assert cutoff==datetime(2026,10,30,21,tzinfo=timezone.utc)
 clock=m.check_submit_clock
 for t in [cutoff-timedelta(days=1),cutoff-timedelta(microseconds=1),cutoff.replace(tzinfo=None)]:
  try:clock(t)
  except ValueError:pass
  else:raise AssertionError('Early/naive time must reject')
 for t in [cutoff,cutoff+timedelta(microseconds=1),cutoff.astimezone(timezone(timedelta(hours=-4)))]:clock(t)
 calls=[];read=m.checked_readiness;submit=m.submission
 m.checked_readiness=lambda *args:calls.append('readiness');m.submission=lambda **kw:calls.append('native_constructor')
 m.check_submit_clock=lambda:clock(cutoff-timedelta(microseconds=1))
 try:
  try:m.request('A9',{},'absent', 'a'*64)
  except ValueError as error:assert 'held until' in str(error)
  else:raise AssertionError('Early request must reject')
  assert calls==[]
 finally:m.checked_readiness=read;m.submission=submit
 # Simulated post-21 time proves construction only; not a current-time launch claim.
 m.check_submit_clock=lambda:clock(cutoff)
 creds={'CW_KEY_ID':'fixture-only','CW_KEY_SECRET':'fixture-only'}
 req=m.request('A9',creds,plan['actual_readiness']['path'],plan['actual_readiness']['sha256'])
 inspect.signature(IrisClient.submit).bind(None,**req)
 assert req['name']==plan['targets']['A9']['root_name'] and req['timeout'].to_seconds()==176400 and req['max_retries_failure']==req['max_retries_preemption']==0
 assert not req['entrypoint'].workdir_files and plan['targets']['A9']['output'] in req['entrypoint'].command[-1]
 audit=m.checked_readiness('A9',plan['actual_readiness']['path'],plan['actual_readiness']['sha256']);assert audit=='8301046f213279b9f8480f21a630549d9e3e76bcb7bdb365c72981d31d435da7'
 child=owner.serve_submission('A9',req['name'],creds,audit,plan['actual_readiness']['sha256']);inspect.signature(IrisClient.submit).bind(None,**child)
 assert child['timeout'].to_seconds()==43200 and child['max_retries_failure']==child['max_retries_preemption']==0
 for item in [req,child]:
  command=item['entrypoint'].command[-1];assert audit in command and plan['actual_readiness']['sha256'] in command
 oldplan=json.loads((parent/'launch-plan.json').read_bytes())
 for key,value in oldplan['targets']['A9'].items():
  if key not in ['root_name','output']:assert plan['targets']['A9'][key]==value
 assert owner.service_config('A9',req['name'])==owner.service_config('A9',oldplan['targets']['A9']['root_name'])
 bundle=create_workspace_zip(Path(plan['workspace']));digest=hashlib.sha256(bundle).hexdigest();assert digest=='bd4004494d932971e1b36919248064922823f17bad5c63c482487dbedb9b96cd'
 with zipfile.ZipFile(io.BytesIO(bundle)) as z:members=len(z.namelist())
 proof={'scope':'CPU-only method parity/time gate + actual same-target readiness/native constructor/bundle; simulated future time is NOT current launch permission','plan_sha256':sha(HERE/'launch-plan.json'),'source_pin':plan['code_pin'],'native_bundle_sha256':digest,'native_bundle_bytes':len(bundle),'native_bundle_members':members,'unchanged_readiness_method_and_request_body_AST_except_leading_clock_guard':'PASS','pre_cutoff_request_zero_readiness_or_native_construction':'PASS','early_naive_reject_exact_cutoff_and_offset_equivalence':'PASS','actual_same_target_502_readiness_and_propagated_audit_digest':'PASS','same_service_config_and_all_target_fields_except_root_output':'PASS','native_parent_command':req['entrypoint'].command,'native_child_command':child['entrypoint'].command,'parent_R2_all132_and_extractedCLI_proof':{'path':str(parent/'native-one-binding-proof.json'),'sha256':sha(parent/'native-one-binding-proof.json')},'generation_template_runtime_source_changes':0,'model_storage_Iris_calls':0,'submit_not_before_utc':plan['not_before_utc'],'no_automatic_further_attempt':True}
 p=HERE/'clock-and-native-binding-proof.json';p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k not in ['native_parent_command','native_child_command']}))
if __name__=='__main__':main()
