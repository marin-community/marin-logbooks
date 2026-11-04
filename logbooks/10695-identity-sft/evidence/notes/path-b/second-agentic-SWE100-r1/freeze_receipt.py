from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;B=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'launch-plan.json').read_text());root=Path(plan['workspace']);proof=json.loads((HERE/'native-twelve-bindings-proof.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
assert len(proof['all12'])==12 and len({x['output_dir'] for x in proof['all12']})==12
for x in proof['all12']:
 t=plan['targets'][x['label']];r=next(y for y in t['runs'] if y['run']==x['run'])
 assert x['output_dir'].startswith(r['records_prefix']+'/')
files=[p for p in HERE.iterdir() if p.is_file() and p.name not in ['review-receipt.json','science-verdict-marten-r1.json','science-reading-marten-r1.md']]
files += [root/p for p in ['experiments/evaluation/held_bfcl.py','config/coreweave.yaml','experiments/evaluation/task238-screens/swebench-verified-100.yaml','experiments/evaluation/launch.py','lib/marin/src/marin/evaluation/harbor/runner.py','lib/marin/src/marin/evaluation/harbor/driver_config.py']]
files += [Path(plan[k]['path']) for k in ['preregistration','cost_choice']]
for t in plan['targets'].values():
 for guard in t['guards'].values():
  if isinstance(guard,dict) and 'local' in guard:files.append(Path(guard['local']))
files += [B/'bfcl-noise-r1/second-agentic-cost-scope-addendum-marten-r1.json',Path('/home/karan/.cache/uv/git-v0/checkouts/58d46c4858ecc44b/38742b0f/src/harbor/agents/installed/mini_swe_agent.py'),Path('/home/karan/.cache/uv/git-v0/checkouts/58d46c4858ecc44b/38742b0f/src/harbor/agents/installed/mini_swe_model.py')]
files=list(dict.fromkeys(files))
receipt=dict(schema_version=1,code_pin=plan['code_pin'],parent_pin=plan['parent_pin'],plan_sha256=sha(HERE/'launch-plan.json'),native_bundle_sha256=proof['bundle_sha256'],source_delta='track exact exercised wrapper functionAST + regionalbucket mapping; standardtask238 science unchanged',submit_authorized=False,model_or_storage_calls_by_genet=0,files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
(HERE/'review-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(sha(HERE/'review-receipt.json'),len(files))
