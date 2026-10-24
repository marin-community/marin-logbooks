import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path('/private/tmp/native-jax-splash-15b-profile-results/float32')
paths={v:ROOT/f'{v}.optimized_hlo.txt' for v in ('legacy','native')}
lines={v:p.read_text().splitlines() for v,p in paths.items()}

def extract(v,a,b):
    return [{'line':i,'text':lines[v][i-1].split('backend_config=')[0].rstrip(', ')} for i in range(a,b+1)]
def need(v,n,*parts):
    s=lines[v][n-1]
    for p in parts:
        assert p in s,(v,n,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

# Assert dataflow, not metadata/name coincidence.
for n,parts in {
2176:('cosine(%maximum.55)',),
2177:('constant(0.0883883461)',),
2178:('broadcast(%constant.484.clone.6)',),
2179:('multiply(%cos.37, %broadcast.750)',),
2180:('sine(%maximum.55)',),
2181:('multiply(%sin.34.clone.3, %broadcast.750)',),
2182:('tuple(%multiply.162, %multiply.108.clone.3, %cos.37, %sin.34.clone.3)',),
3364:('fusion(%broadcast_multiply_fusion.8)','calls=%fused_computation.170.clone.1'),
3542:('get-tuple-element(%broadcast_multiply_fusion.9), index=1',),
3543:('copy-start(%get-tuple-element.1796)',),
3556:('get-tuple-element(%splash_mha_dq_no_residuals.1), index=1',),
3559:('copy-done(%copy-start.18)',),
3560:('fusion(%get-tuple-element.1795, %copy-done.18)','calls=%fused_computation.110.clone.1'),
3561:('get-tuple-element(%broadcast_multiply_fusion.9), index=0',),
3562:('copy-start(%get-tuple-element.1799)',),
3563:('fusion(%multiply_bitcast_fusion.10)','calls=%fused_computation.374.clone.1'),
3564:('get-tuple-element(%fusion.384), index=1',),
3565:('get-tuple-element(%fusion.384), index=0',),
3566:('copy-done(%copy-start.19)',),
3567:('fusion(%get-tuple-element.1797, %get-tuple-element.1798, %get-tuple-element.1795, %copy-done.19)','calls=%fused_computation.109.clone.1'),
2648:('multiply(%param_0.1261, %broadcast.758)',),
2654:('slice(%param_0.1262)','[64:128]'),
2655:('slice(%param_0.1262)','[0:64]'),
2656:('negate(%slice.54.clone.3)',),
2664:('multiply(%param_2.1383, %broadcast.759)',),
2672:('add(%bitcast.469, %maximum.58)',),
}.items():need('native',n,*parts)
for n,parts in {
2660:('cosine(%maximum.45)',),
2661:('sine(%maximum.45)',),
2662:('tuple(%cos.18, %sin.15.clone.3)',),
3222:('constant(0.0883883461)',),
3224:('multiply(%param_0.1193, %eval_jaxpr.119)',),
3808:('calls=%fused_computation.212.clone.1',),
3809:('get-tuple-element(%fusion.359), index=1',),
3811:('get-tuple-element(%fusion.359), index=0',),
3814:('copy(%get-tuple-element.1799)',),
3813:('copy(%get-tuple-element.1800)',),
3959:('get-tuple-element(%splash_mha_dq_no_residuals.1), index=1',),
3963:('fusion(%get-tuple-element.1816)','calls=%fused_computation.176.clone.1'),
3964:('bitcast(%broadcast_multiply_fusion.4)',),
3965:('copy(%bitcast.571)',),
3962:('copy-start(%copy.264)',),
3966:('copy-done(%copy-start.19)',),
3971:('fusion(%copy.269, %copy-done.19)','calls=%fused_computation.147.clone.1'),
3231:('multiply(%param_0.1194, %transpose.414)',),
3974:('fusion(%multiply_bitcast_fusion.9)','calls=%fused_computation.174.clone.1'),
3237:('slice(%param_0.1195)','[0:64]'),
3238:('negate(%slice.129)',),
3239:('slice(%param_0.1195)','[64:128]'),
4001:('copy-start(%copy.269)',),
4002:('copy-start(%copy.265)',),
4010:('get-tuple-element(%slice_negate_fusion.15), index=1',),
4011:('get-tuple-element(%slice_negate_fusion.15), index=0',),
4012:('copy-done(%copy-start)',),
4013:('copy-done(%copy-start.20)',),
4014:('fusion(%fusion.369, %fusion.370, %copy-done, %get-tuple-element.1817, %get-tuple-element.1818, /*index=5*/%copy-done.20, %all-gather.91)','calls=%fused_computation.90.clone.1'),
3273:('fusion(%param_2.1275, %param_3.1001, %param_4.725, %param_5.621)','calls=%fused_computation.142.clone.1.clone.1'),
3253:('multiply(%param_0.1206, %transpose.417)',),
3255:('add(%add_any.170, %bitcast.490)',),
}.items():need('legacy',n,*parts)

report=json.loads((ROOT/'compare.json').read_text())
assert report['provenance']['git_commit']=='36e2a9a97fab22b76dfa5a03964c6600ec5713ac'
assert report['provenance']['source_tree_state']=='clean'
assert report['provenance']['config']['compute_dtype']=='float32'
core=Path('/private/tmp/marin-native-jax-splash-mask-sharding/lib/levanter/src/levanter/grug/attention/_core.py')
assert 'q_ = q_ * (1.0 / math.sqrt(D))' in core.read_text().splitlines()[319]
original_core = subprocess.run(['git','show',report['provenance']['git_commit']+':lib/levanter/src/levanter/grug/attention/_core.py'],cwd='/private/tmp/marin-native-jax-splash-mask-sharding',check=True,capture_output=True).stdout
assert original_core == core.read_bytes()

result={
 'schema':'native_jax_fp32_qscale_static_independent_review.v1',
 'scope':'Read-only static inspection of existing normal timed FP32 Trainer optimized HLO; no hardware run, source edit, acceptance change, or PR mutation.',
 'source_commit':report['provenance']['git_commit'],
 'report':{'path':str(ROOT/'compare.json'),'sha256':sha(ROOT/'compare.json'),'provenance':report['provenance']},
 'hlo_files':{v:{'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size,'line_count':len(lines[v])} for v,p in paths.items()},
 'source_scale':{'path':str(core),'sha256':sha(core),'matches_pinned_git_source_bytes':True,'line':320,'text':core.read_text().splitlines()[319]},
 'conclusions':{
  'native_folds_scale_into_rope_coefficients_before_multiplying_raw_dq':True,
  'legacy_materializes_dq_times_scale_before_both_rope_branches':True,
  'same_scale_literal':'f32(0.0883883461), nearest FP32 representation of 1/sqrt(128)',
  'native_pseudocode':'sc = f32(cos(theta)*scale); ss = f32(sin(theta)*scale); qbar = raw_dq*sc + rotate_transpose(raw_dq*ss), with FP32 operation results and backend fusion details as shown in HLO',
  'legacy_pseudocode':'scaled_dq = materialize_f32(raw_dq*scale); qbar = scaled_dq*cos(theta) + rotate_transpose(scaled_dq*sin(theta)), with FP32 operation results and backend fusion details as shown in HLO',
  'new_experiment_hypothesis':'A forward optimization barrier immediately before query scaling, also enabled for FP32, may preserve the reverse scaled-dQ boundary and align native FP32 lowering with legacy. Validate candidate optimized HLO and actual frozen-value/VJP outputs before any full Trainer gate/performance claim.',
 },
 'native_operand_chain':[
  {'stage':'trig_and_scaled_trig_materialization','lines':extract('native',2170,2183)},
  {'stage':'while_call','lines':extract('native',3364,3364)},
  {'stage':'sin_coefficient','lines':extract('native',3542,3543)},
  {'stage':'raw_dq_producer','lines':extract('native',3553,3559)},
  {'stage':'sin_branch_and_cos_branch','lines':extract('native',3560,3567)},
  {'stage':'sin_multiply','lines':extract('native',2644,2650)},
  {'stage':'sin_rotate_transpose','lines':extract('native',2652,2658)},
  {'stage':'cos_multiply_and_branch_add','lines':extract('native',2660,2673)},
 ],
 'legacy_operand_chain':[
  {'stage':'unscaled_trig_materialization','lines':extract('legacy',2654,2663)},
  {'stage':'trig_tuple_indices','lines':extract('legacy',3808,3814)},
  {'stage':'isolated_scale_computation','lines':extract('legacy',3220,3225)},
  {'stage':'raw_dq_and_scale_call_and_materialization','lines':extract('legacy',3956,3966)},
  {'stage':'sin_branch','lines':[ *extract('legacy',3971,3971),*extract('legacy',3227,3240),*extract('legacy',3974,3974)]},
  {'stage':'cos_branch_and_projection_call','lines':[ *extract('legacy',4001,4002),*extract('legacy',4010,4014),*extract('legacy',3269,3277),*extract('legacy',3243,3256)]},
 ],
 'limits':[
  'This proves an optimized-HLO operand association difference in the actual normal FP32 Trainer, not merely an op-name coincidence. It does not identify the first differing runtime intermediate.',
  'No common runtime raw Splash dQ/cos/sin arrays were captured for a controlled two-order replay in this static task, so it does not quantify the numerical effect or show that all native/legacy operands match before this point.',
  'The artifact has separate materialized fusion outputs around the relevant multiplications; final backend code and further fusion/rounding details are not independently disassembled here.',
  'FP32 first-step gradient and Adam gates passed in the report, whereas the loss trajectory failed; observed reassociation is a concrete hypothesis, not causal proof of the 20-step loss drift or near-zero-coordinate Adam amplification.',
  'The currently pinned 8aa8674d candidate enables a barrier only for BF16 and therefore cannot test this FP32 hypothesis.',
  'An FP32 barrier experiment must be separate and pinned, retain unchanged approved criteria/model/LR/weights/physical placements, and check full Trainer numerics and timings before adoption.',
  'The fixed FP32 fragment control is corroborating context only; this review establishes the normal Trainer chain directly and does not infer normal behavior from fragment naming.',
 ],
 'assertions_passed':True,
}
# Never copy encoded backend bodies into this diagnostic report.
encoded=json.dumps(result,indent=2)
assert 'TLzv' not in encoded and 'backend_config' not in encoded
out=Path('/private/tmp/native-jax-fp32-qscale-static-independent-review.json')
out.write_text(encoded+'\n')
print(json.dumps({'path':str(out),'size_bytes':out.stat().st_size,'sha256':sha(out),'assertions_passed':True,'hlo_shas':{v:sha(p) for v,p in paths.items()}},indent=2))
