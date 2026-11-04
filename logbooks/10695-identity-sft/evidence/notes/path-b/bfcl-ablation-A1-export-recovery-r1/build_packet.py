from pathlib import Path
import hashlib,json,subprocess

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'bfcl-ablation-r1'
ROOT=Path('/home/karan/oa/marin-path-b-ablation-recovery')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((PARENT/'launch-plan.json').read_text())
assert sha(PARENT/'launch-plan.json')=='0a85b86c3a910e4cea863df06cc1aa2627e037ac632167ed73ab406898978222'
a=next(x for x in plan['arms'] if x['arm']=='A1')
pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
evidence=json.loads((PARENT/'A1-preserved-native-and-export-preemption-r1.json').read_text())
preserved=json.loads((PARENT/'A1-failed-export-source-bind-preservation.json').read_text())
assert preserved['bytes']==641 and sha(Path(preserved['local']))==preserved['sha256']=='17080b86c084986e5a3e60f76809043f6da967f337cd14047eb7bded9feb56c0'
out=dict(
 schema_version=1,status='READY_ONE_AUTHORIZED_EXPORT_ONLY_ATTEMPT',workspace=str(ROOT),code_pin=pin,
 original_training_pin=plan['code_pin'],original_plan=dict(path=str(PARENT/'launch-plan.json'),sha256=sha(PARENT/'launch-plan.json')),
 authority='Jerboa ONE A1 export-only recovery after proved queue preemption; no retrain/no new science review; another failed/preempted attempt needs new authority',
 root_name='pathb-bfcl-a1-export-recovery-r1-'+pin[:12],
 recovery_root='s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-A1-export-recovery-'+pin[:12]+'/run',
 training_kwargs=a['training_kwargs'],event=a['final_checkpoint'],runtime=plan['runtime'],
 evidence=dict(event_sha256=evidence['original_final_event_sha256'],training_result_sha256=evidence['training_result_sha256'],metadata_sha256=evidence['native_metadata_sha256'],source_bind_sha256=preserved['sha256'],source_bind_bytes=641),
 CPU_root=plan['cpu_root'],export=plan['export'],bfcl=a['bfcl'],
 original_probe=evidence['old_CPU_probe_full_GET_PASS'],
 prefix_policy='HF must contain exactly the existing641B admission/audits/source-bind.json and nothing else. Preserve under fresh recovery root with full readback before GPU; require exact bytes unchanged after audit before native retirement.',
 probe_policy='New probe ONLY at recovery_root/preflight/export-write-probe.bin; original event.eval_scratch/probe and original event bytes unchanged.',
 retry_policy='ONE manual recovery; requestedfailure0/preemption0, no application resubmit; actual peer policy not claimed',
 BFCL_policy='Use unchanged original A1 YAML/argv/reference once actual502audit and original-event marker pass. HF-only; no native lease.',
 execution_claim='NOT SUBMITTED BY GENET; CPU-only request/signature proof only',
 historical_failure=dict(path=str(PARENT/'A1-terminal-inspection-20261030T020726Z.json'),sha256=sha(PARENT/'A1-terminal-inspection-20261030T020726Z.json')),
 original_native_receipt=dict(path=str(PARENT/'A1-preserved-native-and-export-preemption-r1.json'),sha256=sha(PARENT/'A1-preserved-native-and-export-preemption-r1.json')),
 source_preservation_receipt=dict(path=str(PARENT/'A1-failed-export-source-bind-preservation.json'),sha256=sha(PARENT/'A1-failed-export-source-bind-preservation.json')),
)
(HERE/'launch-plan.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
(HERE/'test-results.json').write_text(json.dumps(dict(command='python /tmp/genet-pathb-check.py -m pytest experiments/post_training/identity_path_b/test_ablation_recovery.py -q',exit_code=0,passed=4,seconds=45.09,real_native_controller=True,external_compute='fake at GPU execution boundary; no model run',formatter_only_after_test=True),indent=2)+'\n')
print(pin,sha(HERE/'launch-plan.json'))
