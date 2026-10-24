"""Static retained-HLO and installed-JAX barrier-rule review; no compilation."""
import hashlib
import json
from pathlib import Path

root = Path('/private/tmp')
r = root / 'native-jax-bf16-splash-15b-separate-vjp-packed-results'
paths = {variant: {kind: r / f'last-{variant}-prefix.{kind}.txt' for kind in ['hlo', 'stablehlo']} for variant in ['named', 'native']}
texts = {variant: {kind: path.read_text() for kind, path in kinds.items()} for variant, kinds in paths.items()}
private = {}
for variant in ['named', 'native']:
    lines = texts[variant]['stablehlo'].splitlines()
    private[variant] = {}
    for i, line in enumerate(lines):
        if line.startswith('  func.func private @_splash_attention'):
            end = next(j for j in range(i + 1, len(lines)) if lines[j] == '  }')
            name = line.split('@')[1].split('(')[0]
            private[variant][name] = '\n'.join(lines[i:end + 1])
assert private['named'] == private['native']
lax_path = Path('/Users/dlwh/.codex/worktrees/73be/marin/.venv/lib/python3.12/site-packages/jax/_src/lax/lax.py')
lax_text = lax_path.read_text()
assert 'ad.primitive_jvps[optimization_barrier_p] = _opt_barrier_jvp' in lax_text
assert 'ad.primitive_transposes[optimization_barrier_p] = _opt_barrier_transpose' in lax_text
assert 'out = iter(optimization_barrier(nz_cts))' in lax_text
report = {
    'scope': 'Independent read-only retained optimizedHLO/StableHLO and installedJAX source audit; no patch, compilation, model execution, hardware or download.',
    'source_commit': '36e2a9a97fab22b76dfa5a03964c6600ec5713ac',
    'retained_runtime_jax_version': json.loads((r / 'provenance.json').read_text())['jax'],
    'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for kinds in paths.values() for path in kinds.values()},
    'dataflow_verified': {
        'stablehlo_named': 'Lines145-155: Splash backward returns dq as115#0; reshape116; BF16scalar0.08837890625 multiply118; transpose121; inverseRoPE multiplies131/138 at167/176.',
        'stablehlo_native': 'Lines218-234: Splash backward returns dq as197#0; shardingconstraint198; sameBF16scalar multiply203; transpose209/constraint210; inverseRoPE multiplies231/235 at256/260.',
        'optimized_named': 'Entry1334-1340 obtains rawSplashdq index1 then comp42 rounds dq*scale into BF16 at800. Entry1342-1348 reshapes/copies result into comp39 multiplying by BF16sin and rounding; inverseRoPE cos branch follows. Scale round precedes trig products.',
        'optimized_native': 'Comp55 lines244-265 computes BF16(scale*cos) and BF16(scale*sin) from BF16trig/scale via F32multiply+reduceprecision8/7+convert. Entry1050/1072 obtains tupleindices2/0;1071 and1078 multiply RAW Splashdq with these scaledtrig arrays through comp23(555-565)/comp22(578-600). No prior BF16(dq*scale) buffer exists on those branches.',
        'rounding_distinction': 'LetR meanBF16rounding and s=0.08837890625. Named inverseRoPE branch usesR(R(dq*s)*trig); native usesR(dq*R(s*trig)). Both have the same real-arithmetic derivative; BF16 rounding differs.',
    },
    'private_splash_functions_byte_identical': {name: {'bytes': len(text.encode()), 'sha256': hashlib.sha256(text.encode()).hexdigest()} for name, text in private['named'].items()},
    'barrier_source': {'path': str(lax_path), 'sha256': hashlib.sha256(lax_path.read_bytes()).hexdigest(), 'lowering_lines': [9947, 9958], 'jvp_lines': [9964, 9973], 'transpose_lines': [9975, 9981], 'facts': ['Lowers toStableHLO OptimizationBarrierOp and preserves explicit sharding in types.', 'JVP barriers primals and nonzero tangents.', 'Transpose barriers nonzero cotangents; zero cotangents pass through.', 'Publicdocstring line9906 says no derivative/batching rules, but installed0.11.1 implementation explicitly registers both plusbatcher. Actualsource is the relevant evidence.']},
    'candidate_placement': {'file': 'lib/levanter/src/levanter/grug/attention/_core.py', 'function': '_tpu_splash_attention', 'before_line': 320, 'candidate': 'q_ = jax.lax.optimization_barrier(q_)\nq_ = q_ * (1.0 / math.sqrt(D))', 'reverse_effect': 'Forward barrier on postRoPE pre-scale query becomes reverse barrier on already-scaled dq before inverseRoPE. This is the relevant round boundary.', 'incorrect_alternative': 'Barrier only after forward query scaling separates Splash from scale, but reverse barrier is then before dq*scale and does not separate scale from inverseRoPE.'},
    'verdict': 'Concrete optimized-dataflow rounding difference independently verified; pre-scale query barrier is a narrowly justified candidate for preserving source-operation rounding order. No causal fix or production correctness claim yet.',
    'required_followup_limits': [
        'Candidate reverseStableHLO and optimizedTPUHLO must retain barrier/scaled-dq-before-trig boundary; sourceAD identity alone does not prove backend honors it.',
        'Compare same frozenoperand/sourceguards first. A barrier also changes forward fusion, copies/memory and scheduling; require forward checks and do not attribute every changed result solely to one reassociation without new evidence.',
        'Privatekernel bodies are byteidentical, not a proof of identical runtime kerneloperand values, physicalcallcontexts or every instruction.',
        'Single-prefix evidence does not explain all wholeTrainer gradients/loss trajectory or establish earliest runtime divergence. Other BF16/RMS/MLP reductions and contexts can still differ.',
        'Do not weaken model-level gates or infer accepted performance. Barrier can reduce fusion/throughput; both numerical and timing consequences require matched validation.',
        'Installed/runtimeJAX0.11.1 rule verified; dependency/version changes require recheckingAD support.'
    ],
}
path = root / 'native-jax-bf16-qscale-reassociation-independent-review.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(path, path.stat().st_size)
print(json.dumps(report['private_splash_functions_byte_identical'], indent=2))
