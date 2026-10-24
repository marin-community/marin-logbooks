from pathlib import Path
import hashlib,json,re

base=Path('/private/tmp/native-jax-bf16-splash-15b-separate-vjp-packed-results')
prov=json.loads((base/'provenance.json').read_text());controls=json.loads((base/'controls.json').read_text());outcome=json.loads((base/'case-outcomes.json').read_text())
stages={s:json.loads((base/f'last-{s}.json').read_text()) for s in ['full','prefix','tail','composed']}
order=stages['full']['named']['result_order']
assert prov['capture_controls_exact'] and prov['source_operand_hashes_exact'] and controls['attribution_valid']
assert all(controls['single_jit_full_vs_composed_exact'].values())
assert all(stages[s]['common_bf16_operands_fp32_masters_and_physical_specs_exact'] for s in stages)
assert all(r['exact'] for r in stages['tail']['comparisons']['named_vs_native'])
summary={}
for stage,data in stages.items():
 comparisons={key:{name:row for name,row in zip(order,rows,strict=True)} for key,rows in data['comparisons'].items()}
 summary[stage]={'comparisons':comparisons,'named_native_nonexact_names':[name for name,row in comparisons['named_vs_native'].items() if not row['exact']],
  'input_and_master_and_cotangent_specs_hashes_match':data['common_bf16_operands_fp32_masters_and_physical_specs_exact']}
private={}
for stage in ['full','prefix']:
 private[stage]={}
 for variant in ['named','native']:
  source=(base/f'last-{variant}-{stage}.stablehlo.txt').read_text()
  private[stage][variant]={m.group(1):{'sha256':hashlib.sha256(m.group(0).encode()).hexdigest(),'bytes':len(m.group(0).encode())} for m in re.finditer(r'^  func.func private (@_splash_attention\w*).*?(?=^  func.func private |^\}\s*$)',source,re.M|re.S)}
 assert private[stage]['named']==private[stage]['native']
same={}
for name in ['input_vjp','input_layernorm.weight','self_attn.q_proj.weight']:
 i=order.index(name);same[name]={v:{'full_sha256':stages['full'][v]['results'][i]['sha256'],'prefix_sha256':stages['prefix'][v]['results'][i]['sha256'],'exact_hash_match':stages['full'][v]['results'][i]['sha256']==stages['prefix'][v]['results'][i]['sha256']} for v in ['named','native','fp32']}
source=Path('/private/tmp/native-jax-splash-15b-profile-results/bfloat16/compare.json')
old=json.loads(source.read_text())['parity']['gradients']['tensors']
fullmodel_compare={name:{'full_trainer_diagnostic_gradient':old['model.layers.27.'+name], 'separate_full_common_operand_vjp':summary['full']['comparisons']['named_vs_native'][name]} for name in order[2:]}
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in base.iterdir() if p.is_file()}
report={
 'schema':'native-jax-splash-15b-separate-vjp-packed-numerical-analysis-v1',
 'scope':'Read-only analysis of aggregate JSON and retained StableHLO/optimized HLO. No execution, new test, download, source/math/threshold/LR edit or hardware operation.',
 'source_commit':prov['source_commit'],'script_sha256':prov['script_sha256'],'parameter_count':prov['parameter_count'],'layer':prov['layer'],
 'provenance_guards':{key:prov[key] for key in ['parameter_sha256','batch_sha256','attention_selection','variant_matmul_precision','xla_flags','capture_controls_exact','source_operand_hashes_exact','source_operand_hashes_vs_original','source_cotangents_nonzero','ordinary_scan_vs_captured_hidden','suffix_vs_scan_loss','source_scan_vs_replayed_last_block','source_full_vs_staged_forward']},
 'result_order':order,'stage_records':summary,
 'full_vs_composed_exact_all_variants':controls['single_jit_full_vs_composed_exact'],
 'full_vs_fixed_boundary_prefix_hashes':same,
 'fixed_boundary_stitch_controls':{'definition':controls['staged_vs_full_definition'],'comparisons':{v:{name:row for name,row in zip(order,rows,strict=True)} for v,rows in controls['staged_vs_full'].items()}},
 'stablehlo_splash_private_functions_byte_identical_named_native':private,
 'concrete_optimized_hlo_difference':{
  'boundary':'BF16 query scale VJP followed by inverse RoPE, before Q-projection backward GEMMs.',
  'scale':'Both use the same BF16-rounded1/sqrt128 =0.08837890625; neither a different source scale nor a precision-enum mismatch was found.',
  'stablehlo_source_order':{'named_prefix':{'path':str(base/'last-named-prefix.stablehlo.txt'),'scale_lines':[149,155],'inverse_rope_lines':[167,177]},'native_prefix':{'path':str(base/'last-native-prefix.stablehlo.txt'),'scale_lines':[222,235],'inverse_rope_lines':[254,271]},'fact':'Both StableHLO source graphs first multiply raw Splash dQ by BF16 scale, then inverse-RoPE sin/cos products. Named uses grouped Hkv×Hq_per_group axes; native flat Hq plus explicit sharding constraints.'},
  'named_prefix_optimized':{'path':str(base/'last-named-prefix.hlo.txt'),'definition_lines':[793,847],'entry_lines':[1340,1348],
   'sequence':'BF16 dQscaled=roundBF16(dQraw*scale); inverse-RoPE uses separately rounded BF16(dQscaled*cos) and BF16(dQscaled*sin).'},
  'native_prefix_optimized':{'path':str(base/'last-native-prefix.hlo.txt'),'coefficient_definition_lines':[244,265],'coefficient_entry_lines':[978,982],'raw_dq_entry_line':1067,'inverse_rope_definition_lines':[555,599],'inverse_rope_entry_lines':[1050,1078],
   'sequence':'Precompute cosScaled=roundBF16(cos*scale),sinScaled=roundBF16(sin*scale). Multiply raw Splash dQ directly by these scaled coefficients, round products BF16, then combine inverse rotation.'},
  'same_difference_in_full':{'named_path':str(base/'last-named-full.hlo.txt'),'named_scaled_dq_lines':[1214,1221],'native_path':str(base/'last-native-full.hlo.txt'),'native_scaled_coefficients_lines':[280,301]},
  'interpretation':'These are distinct BF16 rounding sequences after optimization, while being equal over real arithmetic. Head64 previous attention replay had power-of-two scale1/8; head128 now has a non-power-of-two BF16 scale. That prior result did not test this noncommuting rounding case.',
  'limits':'Raw Splash dQ, its input/residual arguments, scaled dQ and inverse-RoPE outputs were not saved. Matching kernel code does not prove equal runtime kernel operands/outputs. Therefore the HLO difference is an evidence-backed candidate, not yet a proven explanation of the numerical magnitude or a classified compiler defect.'},
 'reference_interpretation':{
  'reference':'Actual native modules and same Splash backend at FP32 highest, with captured BF16 x/cotangent cast to FP32 and common FP32 masters. It changes compute dtype and fusion as well as precision; it is a diagnostic reference, not exact real arithmetic or a native-vs-named FP32 comparison.',
  'full_forward_named_vs_fp32_relative_l2':summary['full']['comparisons']['named_vs_fp32']['forward']['relative_l2'],
  'full_input_vjp_named_vs_fp32_relative_l2':summary['full']['comparisons']['named_vs_fp32']['input_vjp']['relative_l2'],
  'full_input_vjp_native_vs_fp32_relative_l2':summary['full']['comparisons']['native_vs_fp32']['input_vjp']['relative_l2'],
  'full_q_master_named_vs_fp32_relative_l2':summary['full']['comparisons']['named_vs_fp32']['self_attn.q_proj.weight']['relative_l2'],
  'full_q_master_native_vs_fp32_relative_l2':summary['full']['comparisons']['native_vs_fp32']['self_attn.q_proj.weight']['relative_l2'],
  'finding':'Both BF16 paths show ordinary dtype/precision-sensitive discrepancies against FP32. Neither is uniformly closer: full Q error is about0.9254% named vs0.9277% native. This does not exonerate or identify the particular scale/RoPE reassociation.'},
 'comparison_to_full_trainer_last_layer':fullmodel_compare,
 'conclusions':[
  'The previously invalid duplicate-full capture context has been removed. All retained exact source/normal forward/loss/source-hash controls and33full/composed values pass; this validates the stated common-operand replay boundaries only.',
  'Tail named/native forward,input VJP and4active parameter VJPs are exact (all11returned entries exact;5attention/inputnorm parameter entries are unused zero gradients).',
  'Attention prefix already contains persistent Q-master and input-VJP differences. In both BF16 variants their actual output hashes match full replay exactly; this is stronger than matching cross-variant statistics.',
  'Inputnorm gradients change when isolated from full to fixed-boundary prefix in both implementations. Additional full replay down/K/V discrepancies and absence of the normal full-Trainer up/O discrepancies prove fusion context still changes some gradient details.',
  'No production patch, numerical acceptance or performance/convergence claim is warranted from this diagnostic alone.'],
 'next_bounded_hypothesis':{
  'target':'Same-operand large-shape query-scale/inverse-RoPE backward, holding raw BF16 Splash dQ and BF16 cos/sin fixed.',
  'question':'Does scale-then-round-dQ versus round-scaled-coefficients reproduce the actual post-RoPE/Q-master/input-VJP fingerprint?',
  'minimum_control':'Capture the actual prefix raw Splash dQ and coefficient operands with guard against the proven isolated prefix output/VJP hashes. Evaluate the two observed HLO rounding sequences in separate JIT contexts; compare post-RoPE dQ first, then common-operand Q-projection input/master VJPs. Verify retained HLO actually follows intended rounding boundaries.',
  'distinguish':'This tests a concrete post-optimization rounding-order hypothesis on actual1.53B head128 inputs, not the rejected FP32 full-model attention-axis patch or completed107M head64 stage replay. Failing capture guards must stop attribution.',
  'status':'Proposal only; no implementation, execution or hardware launch performed.'},
 'artifact_manifest':files,
 'retained_output_status':{'files':len(files),'all_values_finite':outcome['all_values_finite'],'source_common_operand_attribution_valid':outcome['attribution_valid'],'full_trainer_native_attribution_tested':False}}
out=Path('/private/tmp/native-jax-bf16-splash-15b-separate-vjp-packed-numerical-analysis.json');out.write_text(json.dumps(report,indent=2)+'\n')
md='''# Large BF16 Splash separate-context VJP analysis

The separate capture passes every exact source guard and all33full/composed comparisons. Full named/native forward outputs are exact. The MLP tail is also exact on the common named residual/cotangent, including input VJP and its4active parameter gradients. The persistent difference is already present in the attention prefix.

| Common-operand named/native result | Max absolute error | Relative L2 | Different elements |
| --- | ---: | ---: | ---: |
| Full and prefix input VJP |3.72529e-9 |0.0588515% |1,650,899/33,554,432 |
| Full and prefix Q-master VJP |4.76837e-7 |0.457671% |2,456,792/4,194,304 |
| Full inputnorm-master VJP |9.53674e-7 |0.536624% |1,253/2,048 |
| Prefix inputnorm-master VJP |9.53674e-7 |0.565345% |1,223/2,048 |

Within each BF16 implementation, full and prefix input/Q VJP **hashes** match exactly. The inputnorm hashes do not match across these contexts. Fixed named r/ct_r substitution remains a distinct control: it must not be described as a pure fusion test. Full replay also produces tiny down/K/V differences where the normal full-Trainer record was exact, while its up/O gradients are exact where normal training differed. The replay has not reproduced every normal training cotangent/fusion detail.

## Concrete optimized rounding-order candidate

Both retained StableHLO graphs multiply the raw Splash dQ by the same BF16 scale(1/sqrt128 rounded to0.08837890625), then apply inverse RoPE. The shared private Splash forward and backward functions are byte-identical between named/native. Runtime kernel operands and dQ outputs were not materialized, so kernel identity does not prove equal internal values.

The optimized prefix HLO differs:

- Named `last-named-prefix.hlo.txt:793–800` rounds `dQraw*scale` toBF16 first. Its inverse RoPE then multiplies that rounded result by ordinary sin/cos(`803–847`; entry1340).
- Native `last-native-prefix.hlo.txt:244–265` precomputes and rounds `sin*scale` and `cos*scale` toBF16. The raw Splash dQ then multiplies these already-scaled coefficients(`555–599`; entry978,1050–1078).

The same distinction appears in full optimized HLO(named1214–1221; native280–301). These sequences agree over real arithmetic but change BF16 double-rounding sites. The earlier107M attention replay used head64 and an exact power-of-two scale1/8; it did not test the new head128 non-power-of-two scale. No source scalar mismatch or precision-enum mismatch was found.

This is a concrete post-StableHLO localization hypothesis. The artifacts do not yet prove it explains the observed error magnitude, locate which optimizer pass changed the sequence, or classify a compiler defect. A production patch is premature.

The matched FP32-highest Splash control is a different-compute-dtype diagnostic reference. Q-master relative error against it is0.9254% for named and0.9277% for native; neither path is uniformly closer across all outputs. This is not exact-real-arithmetic evidence or a new accepted training result.

## Smallest next boundary

Hold actual large-prefix raw BF16 Splash dQ and BF16 cos/sin fixed. Compare the two retained optimized rounding sequences in separate JITs, checking post-RoPE dQ before common-operand Q-projection input/master VJPs. Retain exact isolated-prefix/capture hashes and source controls, and inspect HLO to verify rounding boundaries survived compilation. This tests a specific new head128 rounding-order hypothesis; it does not repeat the rejected full-model attention-axis patch. No new control has been implemented or launched.

Evidence: `native-jax-bf16-splash-15b-separate-vjp-packed-numerical-analysis.json` retains all numerical rows, source guards, HLO pointers/hashes, reference comparisons and artifact manifest. No source/math/threshold/LR change, experiment, remote query or download was performed for this audit. No full-Trainer identity, performance or convergence claim follows.
'''
Path('/private/tmp/native-jax-bf16-splash-15b-separate-vjp-packed-numerical-analysis.md').write_text(md)
print(json.dumps({'output':str(out),'bytes':out.stat().st_size,'md_bytes':len(md.encode()),'source_guard_exact':True,'tail_exact':True,'full_prefix_q_input_hash_match_both_bf16':all(same[name][v]['exact_hash_match'] for name in ['input_vjp','self_attn.q_proj.weight'] for v in ['named','native']),'all_files':len(files)},indent=2))
