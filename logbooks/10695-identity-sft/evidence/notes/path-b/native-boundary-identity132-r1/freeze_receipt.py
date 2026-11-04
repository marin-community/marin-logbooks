from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;B=HERE.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'launch-plan.json').read_text());ROOT=Path(plan['workspace']);proof=json.loads((HERE/'native-three-bindings-proof.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==plan['code_pin']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
assert len(proof['targets'])==3 and proof['original_default_generate_AST_exact_except_injected_factory']
files=[p for p in HERE.iterdir() if p.is_file() and p.name not in ['review-receipt.json','science-verdict-marten-r1.json','science-reading-marten-r1.md']]
files += [p for p in (ROOT/'experiments/post_training/identity_path_b/native_boundary_scope').iterdir() if p.is_file() and p.suffix in ['.py','.json']]
files += [ROOT/'experiments/post_training/identity_path_b'/x for x in ['identity_scope/run.py','identity_scope/payload.zip','identity_scope/source_baseline.py','trend_serve.py','async_launch.py','replay_serve.py']]
files += [ROOT/'experiments/post_training/identity_t2/serve.py',ROOT/'experiments/post_training/identity_t2/capture.py']
files += [Path(plan[k]['path']) for k in ['preregistration','native_prefix_equivalence']]
files += [B/'bfcl-ablation-r1/A3/path-b-checkpoint-event.json',B/'bfcl-ablation-r1/A3/path-b-storage-audit.json',B/'ablation-a8-r1/A8a/actual-metadata/event.json',B/'ablation-a8-r1/A8a/actual-metadata/audit.json']
assert len(files)==len(set(files))
out=dict(schema_version=1,code_pin=plan['code_pin'],parent_pin=plan['parent_pin'],plan_sha256=sha(HERE/'launch-plan.json'),native_bundle_sha256=proof['native_bundle_sha256'],source_delta='Named native-boundary screen; originalprefix/template/payload/defaultgenerate unchanged, injected Session adds onlytwo generationflags plus nativeprefix metadata guards; no P1',status='AUTHOR_PROOFS_PASS_NARROW_BINDING_READ_PENDING',submit_authorized=False,model_GPU_remote_calls_by_genet=0,files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files})
(HERE/'review-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(sha(HERE/'review-receipt.json'),len(files))
