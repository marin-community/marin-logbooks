from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;B=HERE.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((HERE/'launch-plan.json').read_text());root=Path(plan['workspace']);proof=json.loads((HERE/'native-two-bindings-proof.json').read_text());parent=plan['parent_pin']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==plan['code_pin'] and not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
changed=subprocess.check_output(['git','diff','--name-only',parent,plan['code_pin']],cwd=root,text=True).splitlines();assert changed==['experiments/post_training/identity_path_b/native_boundary_scope/owner.py','experiments/post_training/identity_path_b/native_boundary_scope/targets.json']
p=root/changed[0];old=subprocess.check_output(['git','show',parent+':'+changed[0]],cwd=root,text=True);import re
assert re.sub(r'TARGETS_SHA256 = "[0-9a-f]{64}"','TARGETS_SHA256 = "BOUND"',old)==re.sub(r'TARGETS_SHA256 = "[0-9a-f]{64}"','TARGETS_SHA256 = "BOUND"',p.read_text())
files=[p for p in HERE.iterdir() if p.is_file() and p.name!='review-receipt.json']
files += [p for p in (root/'experiments/post_training/identity_path_b/native_boundary_scope').iterdir() if p.is_file() and p.suffix in ['.py','.json']]
files += [root/'experiments/post_training/identity_path_b'/x for x in ['identity_scope/run.py','identity_scope/payload.zip','identity_scope/source_baseline.py','trend_serve.py','async_launch.py','replay_serve.py']]
files += [root/'experiments/post_training/identity_t2/serve.py',root/'experiments/post_training/identity_t2/capture.py']
files += [Path(plan[k]['path']) for k in ['preregistration','native_prefix_equivalence','identity_scope_addendum']]
files += [B/'native-boundary-a9-a10-r1/review-receipt.json',B/'native-boundary-a9-a10-r1/science-verdict-marten-r1.json',B/'write-recovery-a9-a10-r1/review-receipt.json',B/'write-recovery-a9-a10-r1/launch-plan.json']
files += [B/'ablation-a8-r1/A8a/actual-metadata/audit.json',B/'ablation-a8-r1/A8a/actual-metadata/model.safetensors.index.json']
assert len(files)==len(set(files))
out=dict(schema_version=1,code_pin=plan['code_pin'],parent_pin=parent,plan_sha256=sha(HERE/'launch-plan.json'),native_bundle_sha256=proof['native_bundle_sha256'],source_delta='Only recovery model/event destinations and corresponding targetSHA constant; parent native method byte-exact',status='LOCAL_PROOFS_PASS_OWNER_RECOVERY_AUTHORITY_ACTUAL_READINESS_AND_STORAGE_GATE_REQUIRED',review_waiver='Fresh required execution identities are not a new scientific variant; no repeat review for unchanged method',actual_A9_A10_HF_readiness_claimed=False,model_GPU_remote_calls_by_genet=0,files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
(HERE/'review-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(sha(HERE/'review-receipt.json'),len(files))
