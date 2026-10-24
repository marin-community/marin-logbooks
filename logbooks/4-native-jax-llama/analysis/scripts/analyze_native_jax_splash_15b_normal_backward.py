from pathlib import Path
import hashlib,json

root=Path('/private/tmp/marin-native-jax-splash-mask-sharding')
path=Path('/private/tmp/native-jax-splash-15b-profile-results/bfloat16/compare.json')
data=json.loads(path.read_text());p=data['parity'];config=data['provenance']['config']
mu=p['execution_adam_mu']['tensors'];nu=p['execution_adam_nu']['tensors'];grad=p['gradients']['tensors']
assert mu.keys()==nu.keys()==grad.keys() and len(mu)==255
summaries={}
for normal,diagnostic in [('execution_adam_mu','adam_mu'),('execution_adam_nu','adam_nu'),('execution_adam_count','adam_count')]:
 v=p[normal];ts=v['tensors']
 summaries[normal]={'tensor_count':len(ts),'failed_elementwise':{n:r for n,r in ts.items() if not r['allclose']},
  'failed_count':sum(not r['allclose'] for r in ts.values()),'nonzero_difference_count':sum(r['max_abs']>0 for r in ts.values()),
  'zero_numerical_difference_names':[n for n,r in ts.items() if r['max_abs']==0],
  'global_max_abs':v['max_abs'],'allclose':v['allclose'],
  'normal_and_diagnostic_cross_variant_summary_records_identical':v==p[diagnostic]}
rows={name:{'normal_mu':mu[name],'normal_nu':nu[name],'diagnostic_gradient':grad[name],
 'nominal_first_mu_gradient_proxy_difference_max_abs':mu[name]['max_abs']/(1-config['beta1']),
 'nominal_first_mu_gradient_proxy_difference_mean_abs':mu[name]['mean_abs']/(1-config['beta1']),
 'nominal_first_mu_gradient_proxy_relative_l2':mu[name]['relative_l2']} for name in mu}
last={n:r for n,r in rows.items() if n.startswith('model.layers.27.')}
source_files=['lib/levanter/scripts/bench/bench_llama_native.py','lib/levanter/src/levanter/trainer.py','lib/levanter/src/levanter/trainer_state.py']
report={
 'schema':'native-jax-splash-15b-normal-backward-evidence-v1',
 'input_report':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_commit':data['provenance']['git_commit'],'source_tree_state':data['provenance']['source_tree_state'],
  'model_parameter_count':data['provenance']['parameter_count'],'compute_dtype':config['compute_dtype'],'parameters':data['provenance']['parameter_dtype'],'optimizer_moments':data['provenance']['optimizer_moment_dtype'],'optimizer':data['provenance']['optimizer'],'optax_version':data['provenance']['optax'],'beta1':config['beta1'],'beta2':config['beta2'],'atol':config['parity_atol'],'rtol':config['parity_rtol']},
 'source_files':{f:{'path':str(root/f),'sha256':hashlib.sha256((root/f).read_bytes()).hexdigest()} for f in source_files},
 'source_evidence':[
  {'file':source_files[0],'lines':[387,410],'fact':'MP uses FP32 parameter dtype, BF16 compute; optimizer is direct optax.adamw(b1=.9,b2=.999,eps1e-8) without clipping or preceding optimizer transforms.'},
  {'file':source_files[0],'lines':[418,457],'fact':'Initialize state once. Diagnostic step gets a copied state (all array leaves copied), emits gradient/update hook results and Adam moments. Normal state remains unstepped.'},
  {'file':source_files[0],'lines':[459,489],'fact':'Production _jit_train_step_fn_no_hook is separately lowered/compiled, then first invoked on the original fresh state. Snapshot normal next parameters, mu, nu and count to independent host np.array copies before further donation.'},
  {'file':source_files[0],'lines':[490,508],'fact':'Warmups and timed/trajectory steps repeatedly use the same no-hook wrapper. Saved normal first moments predate these steps.'},
  {'file':source_files[1],'lines':[802,864],'fact':'Hook and no-hook wrappers call same _train_step. Gradients computed before state.take_step; _no_hooks only removes run_jit_hooks and hook_infos outputs. Hook outputs can still change compilation/fusion.'},
  {'file':source_files[2],'lines':[254,261],'fact':'Filtered trainable gradients are passed directly to optimizer.update; updates applied afterward. No benchmark gradient clipping/scaling transform precedes Adam.'},
  {'file':source_files[0],'lines':[221,239],'fact':'Report holds cross-variant max/mean/L2/allclose error summaries, not raw moment arrays or normal-vs-diagnostic intra-variant comparisons.'}],
 'retained_normal_hlo_evidence':[
  {'path':str(path.parent/'legacy.optimized_hlo.txt'),'lines':[4498,4518],'fact':'BF16 backward operand converted to F32, mu=.1*g+.9*oldmu, nu=.001*g*g+.999*oldnu in retained normal executable (representative stacked Q state fusion).'},
  {'path':str(path.parent/'native.optimized_hlo.txt'),'lines':[4540,4560],'fact':'Same BF16-to-F32 gradient conversion and F32 .1/.9, .001/.999 moment arithmetic (representative norm-state fusion).'}],
 'normal_moment_summaries':summaries,
 'last_block_all9_mu_and_nu_pass_elementwise':all(r['normal_mu']['allclose'] and r['normal_nu']['allclose'] for r in last.values()),
 'last_block_nonzero_difference_names':[n for n,r in last.items() if r['normal_mu']['max_abs']>0],
 'last_block_records':last,
 'head_and_final_norm_records':{n:rows[n] for n in ['lm_head.weight','model.norm.weight']},
 'embedding_record':rows['model.embed_tokens.weight'],
 'first_step_moment_interpretation':{
  'initial_state':'Optax scale_by_adam initializes count0 and zero mu/nu; benchmark original state is fresh for normal first step.',
  'nominal_formulas':{'mu1':'(1-beta1)*g = .1*g, rounded in FP32','nu1':'(1-beta2)*g**2 = .001*g**2, rounded in FP32'},
  'normal_mu_strict_gate_not_gradient_gate':'Using the same atol1e-5 on mu permits approximately atol1e-4 on g; rtol stays1e-4 under uniform scaling. Thus254/255 mu tensors passing does not establish254/255 gradients meet their original gate.',
  'normal_nu_strict_gate_not_gradient_gate':'Small .001*g^2 values and sign loss mean all255 nu tensors passing is weak gradient evidence. Embedding nu still differs by3.9939914e-6 and1.551115%relativeL2.',
  'scaled_mu_without_jit_change':'Existing normal first-step mu already carries a linearly scaled rounded backward signal. Host-side scaling of normal mu would not require a new JIT or hook; saved summaries can scale max/mean errors by10 and leave L2 approximately unchanged. Raw mu values are not retained here, so cannot compute elementwise recovered-gradient allclose from aggregate errors.',
  'no_exact_recovery_claim':'FP32 multiply/add rounding and possible fusion mean mu/.1 is not an exact reconstruction of the gradient tensor. Pairwise diagnostic/normal summary equality also does not prove within-variant raw moment or gradient identity.'},
 'conclusion':{
  'normal_no_hook_backward_signal_nonidentical':True,
  'what_is_proven':'The untouched production no-hook first step has248/255 numerically different cross-variant first-moment tensors; embedding mu fails the original elementwise check. A backward-derived mismatch exists in the normal executable independently of returning diagnostic gradients. The hook alone cannot explain away the measured native/named divergence.',
  'last_block':'All9 last-block mu/nu tensors pass original elementwise checks. Five have zero numerical difference; up/O/Q/inputnorm are nonzero, consistent in locations and approximate scaling with the diagnostic gradient record. Head and final norm have zero numerical difference.',
  'limits':['No direct normal-gradient tensor is captured; normal gradient exactness, individual mismatching coordinates and allclose classifications cannot be recovered from aggregate mu errors.', 'Equal diagnostic/normal cross-variant error summaries do not prove each variant has equal diagnostic and normal arrays.', 'No earliest runtime backward operator or semantic bug is localized by these moments.', 'Normal scalar count pair has maxdifference0; actual count values are not stored in compare.json (source expects1 after first update).']},
 'all255_tensor_records':rows,
 'scope':'Read-only source/HLO/aggregate report audit. No source edit, JIT change, local/remote numerical experiment, job observation/launch, threshold/LR change or download.'}
out=Path('/private/tmp/native-jax-splash-15b-normal-backward-evidence.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'output':str(out),'bytes':out.stat().st_size,'normal_mu_failed':list(summaries['execution_adam_mu']['failed_elementwise']),'normal_nu_failed':list(summaries['execution_adam_nu']['failed_elementwise']),'normal_mu_nu_nonzero': [summaries[k]['nonzero_difference_count'] for k in ['execution_adam_mu','execution_adam_nu']],'lastblock_nonzero':report['last_block_nonzero_difference_names']},indent=2))
