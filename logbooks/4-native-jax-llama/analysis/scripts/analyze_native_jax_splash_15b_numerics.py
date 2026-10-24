"""Summarize existing bounded benchmark reports; no accelerator or network use."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path('/private/tmp/native-jax-splash-15b-profile-results')
PRIOR=Path('/private/tmp/marin-native-jax-15b-profile-results/float32/compare.json')
SMALL=Path('/private/tmp/native-jax-splash-maskfix-107m-results')
OUT=Path('/private/tmp/native-jax-splash-15b-numerical-analysis')

def load(path):
    return json.loads(path.read_text())

def record(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def tensor_summary(group):
    tensors=group['tensors']
    errors=[{'tensor':key,**value} for key,value in tensors.items()]
    return {'allclose':group['allclose'],'max_abs':group['max_abs'],'tensor_count':len(tensors),
        'elementwise_failed_count':sum(not t['allclose'] for t in tensors.values()),
        'nonexact_count':sum(t['max_abs']!=0 for t in tensors.values()),
        'max_relative_l2':max(t['relative_l2'] for t in tensors.values()),
        'top_relative_l2':sorted(errors,key=lambda t:t['relative_l2'],reverse=True)[:10],
        'elementwise_failed_tensors':[t for t in errors if not t['allclose']],
        'all_tensor_errors':tensors}

def loss_vector(report):
    a=np.asarray(report['variants']['legacy']['measured_step_losses'],dtype=np.float64)
    b=np.asarray(report['variants']['native']['measured_step_losses'],dtype=np.float64)
    assert len(a)==len(b)==20 and np.isfinite(a).all() and np.isfinite(b).all()
    config=report['provenance']['config'];diff=b-a
    close=np.isclose(a,b,atol=config['parity_atol'],rtol=config['parity_rtol'])
    # First untimed production execution plus warmups precede these losses.
    pre_first=1+config['warmup_steps']
    samples=[{'sample':i+1,'state_step_before_update':pre_first+i,'state_step_after_update':pre_first+i+1,'legacy':float(x),'native':float(y),'native_minus_legacy':float(y-x),'elementwise_allclose':bool(close[i])} for i,(x,y) in enumerate(zip(a,b,strict=True))]
    return {'relative_l2_float64_recomputed':float(np.linalg.norm(diff)/np.linalg.norm(a)),
        'relative_l2_report_float32':report['parity']['measured_step_losses']['tensors']['losses']['relative_l2'],
        'max_abs':float(np.max(np.abs(diff))),'mean_abs':float(np.mean(np.abs(diff))),
        'max_abs_sample':samples[int(np.argmax(np.abs(diff)))],
        'elementwise_failed_samples':[s['sample'] for s in samples if not s['elementwise_allclose']],
        'samples':samples,'final_state_step':report['variants']['legacy']['final_step'],
        'final_reported_loss_note':'Last measured loss precedes the final update; final_step=26 is captured before two profiled updates.'}

def backward(report):
    tensors=report['parity']['gradients']['tensors'];layers=report['provenance']['config']['num_layers']
    summaries=[]
    for index in range(layers-1,-1,-1):
        prefix=f'model.layers.{index}.'
        entries=[{'tensor':name,**value} for name,value in tensors.items() if name.startswith(prefix)]
        summaries.append({'layer':index,'tensor_count':len(entries),
            'nonexact_count':sum(t['max_abs']!=0 for t in entries),
            'elementwise_failed_count':sum(not t['allclose'] for t in entries),
            'max_abs':max(t['max_abs'] for t in entries),
            'max_relative_l2':max(t['relative_l2'] for t in entries),
            'errors':entries})
    failed=[s for s in summaries if s['elementwise_failed_count']]
    return {'layer_order':'Descending decoder depth, not a total execution order between parallel projection branches.',
        'tail_head_and_final_norm':{k:tensors[k] for k in ('lm_head.weight','model.norm.weight')},
        'last_layer_errors':summaries[0]['errors'],
        'first_layer_with_any_nonexact_gradient':next((s['layer'] for s in summaries if s['nonexact_count']),None),
        'first_layer_with_any_strict_failed_gradient':failed[0]['layer'] if failed else None,
        'first_strict_failures':[t for t in failed[0]['errors'] if not t['allclose']] if failed else [],
        'by_layer':summaries,
        'onset_limit':'Only parameter-gradient summaries are available. They cannot determine the earliest activation/cotangent mismatch or establish which parallel branch executes first.'}

def report_analysis(path):
    report=load(path);p=report['provenance'];config=p['config']
    assert p['parameter_sha256']==p['native_parameter_sha256']
    assert report['initial_parameter_parity']['max_abs']==0 and report['initial_parameter_parity']['allclose']
    assert all(t['max_abs']==0 for t in report['initial_parameter_parity']['tensors'].values())
    independently_evaluated={}
    for key,comparison in report['parity'].items():
        decision=report['acceptance'][key]
        accepted=all(t['relative_l2']<=decision['relative_l2_limit'] for t in comparison['tensors'].values()) if decision['metric']=='relative_l2' else all(t['allclose'] for t in comparison['tensors'].values())
        assert accepted==decision['accepted']
        independently_evaluated[key]=accepted
    assert report['comparison_valid']==all(independently_evaluated.values())
    assert report['native_speedup'] is None and report['native_median_speedup'] is None
    return {'report':record(path),'source':{k:p.get(k) for k in ('git_commit','source_tree_state','source_bundle_marker','parameter_count','parameter_sha256','native_parameter_sha256','batch_sha256','backend','device_kinds','devices','process_count','mesh','parameter_dtype','optimizer_moment_dtype','jax_default_matmul_precision','attention_backend','attention_entrypoints','attention_splash_max_block_size','xla_flags','jax','jaxlib','equinox','optax','loss_backend')},'config':config,
        'initial_exact_tensor_count':len(report['initial_parameter_parity']['tensors']),
        'comparison_valid':report['comparison_valid'],'speed_ratios':{'mean':report['native_speedup'],'median':report['native_median_speedup']},
        'failed_acceptance_groups':[k for k,v in independently_evaluated.items() if not v],
        'acceptance':report['acceptance'],
        'groups':{key:tensor_summary(group) for key,group in report['parity'].items()},
        'normal_vs_diagnostic_error_summaries_equal':{key:report['parity'][key]==report['parity'][normal] for key,normal in [('adam_mu','execution_adam_mu'),('adam_nu','execution_adam_nu'),('adam_count','execution_adam_count'),('parameters_after_one_step','execution_parameters_after_one_step')]},
        'normal_summary_limit':'Equality of pairwise error summaries does not by itself prove each variant diagnostic and normal state is byte-identical; these records independently demonstrate normal-executable failures.',
        'loss_vector':loss_vector(report),'backward_pattern':backward(report),
        'raw_unaccepted_median_ms':{variant:1000*report['variants'][variant]['timing']['median_step_seconds'] for variant in ('legacy','native')},
        'profiles':{variant:report['variants'][variant].get('profile') for variant in ('legacy','native')}}

reports={dtype:report_analysis(ROOT/dtype/'compare.json') for dtype in ('float32','bfloat16')}
old=load(PRIOR);fp=load(ROOT/'float32/compare.json')
config_deltas={k:{'reference':old['provenance']['config'].get(k),'splash':fp['provenance']['config'].get(k)} for k in sorted(set(old['provenance']['config'])|set(fp['provenance']['config'])) if old['provenance']['config'].get(k)!=fp['provenance']['config'].get(k)}
reference_comparison={'reference_report':record(PRIOR),'reference_source':old['provenance']['git_commit'],
    'initial_weights_exact_same_across_15b_fp32_runs':old['provenance']['parameter_sha256']==fp['provenance']['parameter_sha256'],
    'batch_exact_same_across_15b_fp32_runs':old['provenance']['batch_sha256']==fp['provenance']['batch_sha256'],
    'config_deltas':config_deltas,'matmul_precision':{label:report['provenance']['jax_default_matmul_precision'] for label,report in [('reference',old),('splash',fp)]},
    'reference_gradient_summary':tensor_summary(old['parity']['gradients']),
    'reference_update_summary':tensor_summary(old['parity']['updates']),
    'reference_normal_parameter_summary':tensor_summary(old['parity']['execution_parameters_after_one_step']),
    'reference_loss_vector':loss_vector(old),
    'raw_unaccepted_median_ms':{v:1000*old['variants'][v]['timing']['median_step_seconds'] for v in ('legacy','native')},
    'interpretation':'Splash changes both independent trajectories. FP32 loss-vector L2/max/endpoint differences are smaller, but mean absolute loss difference is slightly larger. This is not a passed numerical gate or a proof of one causal drift mechanism.'}
small={}
for dtype in ('float32','bfloat16'):
    path=SMALL/dtype/'compare.json';report=load(path)
    small[dtype]={'report':record(path),'source_commit':report['provenance']['git_commit'],'config':report['provenance']['config'],'parameter_sha256':report['provenance']['parameter_sha256'],'batch_sha256':report['provenance']['batch_sha256'],'comparison_valid':report['comparison_valid'],'all_groups_zero_max_abs':all(g['max_abs']==0 and all(t['max_abs']==0 for t in g['tensors'].values()) for g in report['parity'].values()),'group_metrics':{k:{'max_abs':g['max_abs'],'failed_tensors':sum(not t['allclose'] for t in g['tensors'].values()),'max_relative_l2':max(t['relative_l2'] for t in g['tensors'].values())} for k,g in report['parity'].items()},'trajectory':loss_vector(report)}
common_path=Path('/private/tmp/native-jax-commonstate-analysis.json')
common=load(common_path)
analysis={'schema':'native-jax-splash-15b-numerical-analysis.v1','scope':'Read-only local bounded diagnostic reports; no live job queries, object downloads, source edits or hardware launches. Physical layout audit is separately owned by parent/peer.',
    'reports':reports,'prior_15b_reference_fp32':reference_comparison,'prior_107m_splash':small,
    'prior_common_state_fp32':{'artifact':record(common_path),'steps':[c['step'] for c in common],
        'step0_selected_reference_residual_fraction':common[0]['selected_reference_residual_fraction'],
        'step25_all_original_elementwise_checks_pass':all(c['elementwise'] for c in common[1]['comparisons'].values()),
        'interpretation':'Already completed reference-attention common-state diagnostic supports near-zero-gradient Adam amplification at selected step0 coordinates and no demonstrated late operator mismatch at common step25. It is neither a BF16 result nor a direct attribution of the new Splash trajectories; do not repeat as a generic retry.'},
    'source_math_evidence':{'native_mlp':'lib/levanter/src/levanter/models/llama_native.py:272-276','named_mlp':'lib/levanter/src/levanter/models/llama.py:287-294','native_block':'lib/levanter/src/levanter/models/llama_native.py:301-308','named_block':'lib/levanter/src/levanter/models/llama.py:331-348','observation':'Both source MLPs implement down(silu(gate(x))*up(x)), with same two residual additions. Linear sharding/lowering and fused derivative context differ; these summaries do not establish an incorrect cast, missing term or semantic mismatch.'},
    'conclusions':[
        '1.53B FP32 Splash fails only the approved20-loss-vector L2 gate: .366196% exceeds .1%; all first-step acceptance groups pass, including no-hook execution parameters/Adam.',
        '1.53B BF16 Splash does not inherit107M bitexact behavior:7of12 acceptance groups fail,26of255 first gradients fail strict checks,248of255 updates/next parameters fail, and20-loss-vector relativeL2 is3.64930%.',
        'BF16 parameter differences already appear in last-layer27 MLP up projection and O/Q/input norm, while head/final norm and several last-layer gradients remain exact. Earliest failing decoder layer walking backward is7(V). This is a localization clue, not identification of the earliest runtime operator.',
        'Normal-executable and diagnostic pairwise state error summaries match for parameters and every Adam component in both dtypes, so the numerical problem is not confined to diagnostic hooks.',
        'No concrete production correctness fix is established. The larger workload simultaneously changes depth,width,MLP,head_dim,vocab,sequence and Splash tile count versus107M; none is isolated causally.'
    ],
    'next_action':{'production_fix_supported':False,'launch_authorized_by_this_analysis':False,
        'bounded_evidence_path':'First use the already-captured BF16 optimized HLO to inspect the last decoder SwiGLU up-branch weight/input VJP, shared residual cotangent addition, and rematerialized derivative fusion. The downstream MLP-up gradient is nonexact before a unique attention-only onset can be inferred.',
        'conditional_diagnostic_if_root_later_selects_it':'A common layer27 block/tail/prefix VJP must use this1.53B shape and Splash source36e, shared actual last-layer input/output cotangent,F32 masters castBF16 inside VJP,and source-capture/full-scan/fused-versus-staged controls. The held source0ac107M reference-attention block replay is not this experiment and must not be launched unchanged.',
        'stop_condition':'If same-operand block/tail remain equal, inspect stacked scan/remat/full-normal execution context before proposing a math change; no criterion or LR changes to force acceptance.'}}
OUT.with_suffix('.json').write_text(json.dumps(analysis,indent=2,sort_keys=True,allow_nan=False)+'\n')

OUT.with_suffix('.md').write_text("# 1.53B Splash numerical assessment\n\nBoth source `36e2a9a97f` comparisons completed, but neither passes the unchanged numerical criteria. No speed ratio is accepted. This analysis reads saved reports only; physical placement verification is a separate audit.\n\nThe workload has 1,531,037,696 parameters: 28 layers, width 2048, MLP 5632, 16 query/4 KV heads, head size 128, vocabulary 65536, batch 8, sequence 2048, data2/model2, on four v5p devices. Masters and Adam moments are FP32. FP32 compute uses highest precision; BF16 compute uses default. Both use `--xla_allow_excess_precision=false`.\n\n| Comparison | FP32 | BF16 |\n|---|---|---|\n| First loss | Exact | Exact |\n| First gradients | All 255 pass; max 6.10948e-7 | 26/255 fail; max .00634765625 |\n| Adam mu | Pass | Embedding fails; all other tensors pass |\n| Adam nu/count | Pass | Pass |\n| Updates | Every tensor passes approved .5% L2; worst .0293347% | 248/255 fail strict checks; worst tensor L2 13.9607% |\n| Diagnostic next parameters | Every tensor passes approved .5% L2; worst .0205737% | 248/255 fail strict checks; worst tensor L2 9.79078% |\n| Normal next parameters/Adam | Same pairwise error summaries as diagnostic | Same pairwise error summaries as diagnostic |\n| 20 measured losses | L2 .366196%, exceeds approved .1% | L2 3.64930%; fails original elementwise checks |\n\nFP32 fails **only** the trajectory, at 3.662 times its approved relative-L2 limit. Maximum absolute loss difference is .0926728. Final measured losses are 7.61323309 legacy / 7.68892336 native. All first-step acceptance groups pass, including parameters and Adam state from the normal executable.\n\nBF16 fails seven of the twelve acceptance groups. Of 255 first-gradient tensors, 248 are nonexact. The worst gradient tensor relative L2 is 1.71984% (layer 0 Q), although that small-gradient tensor passes the absolute elementwise tolerance. Adam mu fails only for the embedding; nu/count pass. The largest update/next-parameter L2 errors are also in the embedding. These are not accepted under the unchanged BF16 criteria.\n\nThe BF16 trajectory has maximum absolute difference .894164 at measured sample 8. Final measured losses are 7.94388533 / 7.85619164. The normal executable independently exhibits parameter/moment errors: all four pairwise state-error summaries exactly equal their diagnostic counterparts. Summary equality does not prove byte identity of each variant's diagnostic and normal state.\n\n## Backward onset\n\nBF16 head and final-norm gradients are exact. In decoder 27, down/gate/post-attention-norm/K/V gradients are exact. **MLP up, O, Q and input norm are already nonexact**, with respective maximum errors 4.76837e-7, 4.76837e-7, 4.76837e-7 and 9.53674e-7. All still pass strict checks. Decoder 26 has every gradient nonexact, and differences spread toward the input. The first strict failing decoder layer encountered walking backward is **layer 7 V**, maximum error 1.14441e-5.\n\nParameter-gradient summaries cannot reveal the earliest activation/cotangent mismatch or order parallel branches. MLP up being nonexact means an attention-only onset is not established. Both source MLPs implement `down(silu(gate(x))*up(x))`; both blocks have the same two residual additions. No missing term or mismatched source cast is demonstrated.\n\n## Existing comparisons\n\nPrior 1.53B reference-attention FP32 source `ad2bc20438` has the same exact initial parameter hash (`d79fb8c3…f453f9a`), batch hash (`d65acfa3…9caa4`), architecture, optimizer, precision and XLA flag. Its 20-loss L2 was .678130% and maximum error .238260; Splash gives .366196%/.0926728. Mean absolute loss difference is slightly larger with Splash (.0143156 versus .0137542), so this is not uniformly smaller error or a passed gate. Backend selection changes both independent trajectories.\n\nThe 107M Splash source `36e2a9a97f` BF16 result was bit-exact in every tensor of all twelve groups and all twenty losses. That result is workload-specific. The larger case changes depth 8→28, width 1024→2048, MLP 2816→5632, head size 64→128, vocabulary 8192→65536 and sequence 512→2048, including attention tiling. No single factor is isolated.\n\nThe completed reference-attention common-state diagnostic at steps 0/25 supports tiny near-zero-gradient differences amplified by first-step Adam at selected coordinates, with all strict comparisons passing at the common step-25 state. This does not prove the new Splash trajectory mechanism and supplies no BF16 evidence. Repeating that generic check is unnecessary.\n\n## Evidence needed next\n\n**No production fix is justified yet.** Inspect the captured BF16 optimized HLO around the final decoder SwiGLU up-branch VJP, shared residual cotangent accumulation and rematerialized fusion. A later common-operand block diagnostic must use the actual 1.53B shape and source `36e2a9` Splash, rather than launching the held 107M/source `0ac` reference script. It should compare decoder-27 full block, MLP tail and attention/residual prefix, with common FP32 masters cast to BF16 inside the VJP and explicit materialized-versus-fused controls. This analysis launches no run.\n\nRaw unaccepted median timings, for provenance: FP32 legacy/native 1032.478/1035.345 ms; BF16 286.208/277.658 ms. No ratios are calculated. Longer training and convergence remain unvalidated.\n\nComplete group/per-tensor errors, all twenty losses, layer records, source pins and report hashes are in `/private/tmp/native-jax-splash-15b-numerical-analysis.json`. Analysis generator: `/private/tmp/analyze_native_jax_splash_15b_numerics.py`.\n")
print(json.dumps({'json':str(OUT.with_suffix('.json')),'markdown':str(OUT.with_suffix('.md')),'failed_groups':{k:v['failed_acceptance_groups'] for k,v in reports.items()}},indent=2))
