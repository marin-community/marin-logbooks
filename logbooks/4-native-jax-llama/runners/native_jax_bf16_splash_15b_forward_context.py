"""Isolate the failed named last-block replay across compiled graph contexts."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import haliax as hax

sys.path.insert(0, str(Path(__file__).parent))
import native_jax_bf16_splash_15b_block_replay as frozen

prior = frozen.prior
SOURCE = frozen.SOURCE
FROZEN_BLOCK_SHA = '858f1885bf452b883d7e1cbaeec11c3beff3628bf6f674aab9789860853e5693'
CAPTURE_HELPER_SHA = '365e8edbf32323e64528953e0122cdcc119da35ac93aaa2a8fa09f703d8f9615'
ORIGINAL_INPUT_SHA = '610eaa4d595f6f112f77a507acf91b208930927bbe9c6d00582736d3ef5937ad'
ORIGINAL_OUTPUT_SHA = 'ea3379fa53134daa136c29f4f3b7987a710a1e97a7e081da65a194c195e3fc9b'
ORIGINAL_ERROR = {'max_abs': 4.19921875, 'relative_l2': 0.16780116533211895, 'different_elements': 33193581, 'elements': 33554432}
HLO_FILE_LIMIT = 4 * 1024 * 1024
HLO_TOTAL_LIMIT = 56 * 1024 * 1024


def definition(tree, name):
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f'Expected one function definition for {name}')
    return matches[0]


def ast_proof():
    original = ast.parse(Path(frozen.__file__).read_text())
    current = ast.parse(Path(__file__).read_text())
    result = {}
    for name in ('scanned_capture', 'capture_last'):
        left, right = definition(original, name), definition(current, name)
        a, b = ast.dump(left, include_attributes=False), ast.dump(right, include_attributes=False)
        result[name] = {'identical_ast_including_decorator_arguments_body_and_returns': a == b,
            'ast_sha256': hashlib.sha256(a.encode()).hexdigest()}
        if a != b:
            raise ValueError(f'Original {name} function graph definition changed')
    return result


def statistics(value):
    named_axes = None
    if isinstance(value, hax.NamedArray):
        named_axes = [{'name': axis.name, 'size': axis.size} for axis in value.axes]
        value = value.array
    array = np.array(jax.device_get(value), copy=True)
    float_array = array.astype(np.float32)
    if not np.isfinite(float_array).all():
        raise ValueError('Nonfinite diagnostic result')
    digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    digest.update(array.tobytes())
    sharding = value.sharding
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'sha256': digest.hexdigest(), 'named_axes': named_axes,
        'max_abs': float(np.max(np.abs(float_array))), 'rms': float(np.sqrt(np.mean(float_array * float_array))),
        'sharding': str(sharding), 'partition_spec': str(sharding.spec),
        'addressable_shards': [{'device': str(shard.device), 'process': shard.device.process_index,
            'index': str(shard.index), 'local_shape': list(shard.data.shape)} for shard in value.addressable_shards]}


class Capture:
    def __init__(self, output):
        self.output = output
        self.hlo_bytes = 0
        self.records = {}

    def run(self, label, fn, args, result_names, **kwargs):
        lowered = fn.lower(*args, **kwargs)
        compiled = lowered.compile()
        # Save both compiler stages before executing or testing result controls.
        hlo = {}
        for suffix, data in [('stablehlo', lowered.as_text().encode()), ('hlo', compiled.compiled.as_text().encode())]:
            keep = min(len(data), HLO_FILE_LIMIT, HLO_TOTAL_LIMIT - self.hlo_bytes)
            if keep <= 0:
                raise ValueError('Diagnostic HLO cap exceeded')
            self.hlo_bytes += keep
            filename = f'{label}.{suffix}.txt'
            (self.output / filename).write_bytes(data[:keep])
            hlo[suffix] = {'file': filename, 'full_bytes': len(data), 'saved_bytes': keep,
                'sha256_full': hashlib.sha256(data).hexdigest(), 'sha256_saved': hashlib.sha256(data[:keep]).hexdigest(),
                'truncated': keep < len(data)}
        start = time.perf_counter()
        result = compiled(*args, **kwargs)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        values = result if isinstance(result, tuple) else (result,)
        if len(values) != len(result_names):
            raise ValueError('Result labels do not describe every returned buffer')
        record = {'hlo': hlo, 'result_order': result_names, 'results': {name: statistics(value) for name, value in zip(result_names, values, strict=True)},
            'matmul_precision': str(jax.config.jax_default_matmul_precision), 'execution_seconds_diagnostic_only': elapsed,
            'all_returned_buffers_blocked_read_and_finite': True, 'saved_hlo_before_execution_and_checks': True}
        prior.write_json(self.output / f'{label}.json', record)
        self.records[label] = record
        return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--expected-backend', choices=('cpu', 'tpu'), required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get('IRIS_OUTPUT_DIR', '/tmp/native-splash-forward-context')))
    args = parser.parse_args()
    if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != FROZEN_BLOCK_SHA:
        raise ValueError('Frozen original block diagnostic changed')
    if hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest() != CAPTURE_HELPER_SHA:
        raise ValueError('Frozen attention helper changed')
    proof = ast_proof()
    if os.environ.get('XLA_FLAGS') != '--xla_allow_excess_precision=false' or os.environ.get('SOURCE_COMMIT', SOURCE) != SOURCE:
        raise ValueError('Expected exact source pin and XLA flags')
    if jax.process_count() != 1 or jax.device_count() != 4 or jax.local_device_count() != 4 or jax.default_backend() != args.expected_backend:
        raise ValueError('Expected one process/four devices/backend')
    if not args.small and (args.expected_backend != 'tpu' or any('v5' not in d.device_kind.lower() or 'lite' in d.device_kind.lower() for d in jax.devices())):
        raise ValueError('Actual case requires v5p')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a fresh empty output directory')
    jax.config.update('jax_default_matmul_precision', 'default')
    config = frozen.configuration(args.small)
    base = prior.Replay(config, output)
    mesh = base.meshes['named']
    capture = Capture(output)
    tokens_host = np.random.default_rng(config.seed + 2).integers(0, config.vocab_size, (config.batch_size, config.seq_len), dtype=np.int32)
    loss_weight_host = np.ones_like(tokens_host, dtype=np.float32)
    loss_weight_host[:, -1] = 0
    with jax.set_mesh(mesh), hax.axis_mapping(prior.bench._parameter_mapping()):
        master = hax.named_jit(prior.bench._legacy_config(config).build, axis_resources=prior.bench._parameter_mapping())(hax.Axis('vocab', config.vocab_size), key=jax.random.PRNGKey(config.seed))
        parameter_count = sum(int(np.prod(a.shape)) for a in jax.tree.leaves(master) if eqx.is_inexact_array(a))
        parameter_hash = prior.bench._tensor_digest(prior.bench._canonical_parameters(master))
        batch_hash = prior.bench._tensor_digest({'tokens': tokens_host, 'loss_weight': loss_weight_host})
        if not args.small and (parameter_count != frozen.LARGE_PARAMETER_COUNT or parameter_hash != frozen.LARGE_PARAMETER_SHA or batch_hash != frozen.LARGE_BATCH_SHA):
            raise ValueError('Actual source model/weights/batch differ from the failed diagnostic')
        model = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if eqx.is_inexact_array(a) else a, master)
        tokens = jax.device_put(tokens_host, NamedSharding(mesh, P('data', None)))
        loss_weight = jax.device_put(loss_weight_host, NamedSharding(mesh, P('data', None)))
        named_tokens = hax.named(tokens, (base.batch, base.pos))
        positions = hax.named(jax.device_put(base.positions, NamedSharding(mesh, P('data', None))), (base.batch, base.pos))
        # D and source scan below have AST-identical definitions to frozen858f.
        @eqx.filter_jit
        def scanned_capture(m, t):
            initial = m.embeddings.embed(t)
            def step(block, carry):
                hidden, _ = carry
                return block(hidden, prior.NamedMask.causal(), pos_ids=positions), hidden
            final, last_input = m.transformer.layers.fold_via(step)((initial, initial))
            axes = (base.batch, base.pos, base.embed)
            return last_input.rearrange(axes).array, final.rearrange(axes).array, m.transformer.norm(final).rearrange(axes).array
        @eqx.filter_jit
        def capture_last(m, x, y):
            block = m.transformer.layers.unstacked()[-1]
            axes = (base.batch, base.pos, base.embed)
            named_x = hax.named(x, axes)
            r = named_x + block.self_attn(block.input_layernorm(named_x), prior.NamedMask.causal(), pos_ids=positions)
            replayed_y = block(named_x, prior.NamedMask.causal(), pos_ids=positions).rearrange(axes).array
            def tail(residual):
                h = hax.named(residual, axes)
                return (h + block.mlp(block.post_attention_layernorm(h))).rearrange(axes).array
            def suffix(hidden):
                h = m.transformer.norm(hax.named(hidden, axes)).rearrange(axes).array
                return prior.bench._shared_loss(h, m.get_lm_head().rearrange(('vocab', 'embed')).array, tokens, loss_weight)
            loss, ct_y = jax.value_and_grad(suffix)(y)
            staged_y, pb_tail = jax.vjp(tail, r.rearrange(axes).array)
            ct_r = pb_tail(ct_y)[0]
            return r.rearrange(axes).array, replayed_y, staged_y, loss, ct_y, ct_r
        @eqx.filter_jit
        def combined_forward_only(m, x):
            block = m.transformer.layers.unstacked()[-1]
            axes = (base.batch, base.pos, base.embed)
            named_x = hax.named(x, axes)
            r = named_x + block.self_attn(block.input_layernorm(named_x), prior.NamedMask.causal(), pos_ids=positions)
            full = block(named_x, prior.NamedMask.causal(), pos_ids=positions)
            composed = r + block.mlp(block.post_attention_layernorm(r))
            return r.rearrange(axes).array, full.rearrange(axes).array, composed.rearrange(axes).array
        @eqx.filter_jit
        def isolated_full(m, x):
            block = m.transformer.layers.unstacked()[-1]
            axes = (base.batch, base.pos, base.embed)
            return block(hax.named(x, axes), prior.NamedMask.causal(), pos_ids=positions).rearrange(axes).array
        @eqx.filter_jit
        def isolated_composed(m, x):
            block = m.transformer.layers.unstacked()[-1]
            axes = (base.batch, base.pos, base.embed)
            named_x = hax.named(x, axes)
            r = named_x + block.self_attn(block.input_layernorm(named_x), prior.NamedMask.causal(), pos_ids=positions)
            return (r + block.mlp(block.post_attention_layernorm(r))).rearrange(axes).array
        normal_hidden_fn = eqx.filter_jit(lambda m, t: m.activations(t, prior.NamedMask.causal(), pos_ids=positions))
        normal_loss_fn = eqx.filter_jit(prior.bench._legacy_loss)
        batch = frozen.LmExample(named_tokens, hax.named(loss_weight, (base.batch, base.pos)), prior.NamedMask.causal())
        with hax.axis_mapping(prior.bench._compute_mapping()):
            x, y, normalized = capture.run('source-scan', scanned_capture, (model, named_tokens), ('last_input', 'last_output', 'normalized_output'))
            # Retain original compilation/execution order for D before perturbations.
            D = capture.run('D-original-capture', capture_last, (model, x, y), ('residual', 'full', 'composed', 'suffix_loss', 'output_cotangent', 'tail_input_cotangent'))
            normal_named_hidden, = capture.run('normal-hidden', normal_hidden_fn, (model, named_tokens), ('normalized_output',))
            normal_hidden = normal_named_hidden.rearrange((base.batch, base.pos, base.embed)).array
            normal_loss, = capture.run('normal-loss', normal_loss_fn, (model, batch), ('loss',), key=None)
            A = capture.run('A-forward-combined', combined_forward_only, (model, x), ('residual', 'full', 'composed'))
            B, = capture.run('B-full-alone', isolated_full, (model, x), ('full',))
            C, = capture.run('C-composed-alone', isolated_composed, (model, x), ('composed',))
    pairs = {'ordinary_normalized_hidden_vs_capture': (normal_hidden, normalized),
        'ordinary_loss_vs_D_suffix': (normal_loss, D[3]),
        'scan_vs_D_full': (y, D[1]), 'scan_vs_D_composed': (y, D[2]),
        'scan_vs_A_full': (y, A[1]), 'scan_vs_A_composed': (y, A[2]),
        'scan_vs_B_full_alone': (y, B), 'scan_vs_C_composed_alone': (y, C),
        'D_full_vs_A_full': (D[1], A[1]), 'D_composed_vs_A_composed': (D[2], A[2]),
        'D_residual_vs_A_residual': (D[0], A[0]), 'B_full_vs_C_composed': (B, C)}
    comparisons = {name: prior.difference(a, b) for name, (a, b) in pairs.items()}
    ordinary_guards = all(comparisons[k]['exact'] for k in ('ordinary_normalized_hidden_vs_capture', 'ordinary_loss_vs_D_suffix'))
    original_operands_match = args.small or (capture.records['source-scan']['results']['last_input']['sha256'] == ORIGINAL_INPUT_SHA and capture.records['source-scan']['results']['last_output']['sha256'] == ORIGINAL_OUTPUT_SHA)
    all_contexts_match_scan = all(comparisons[k]['exact'] for k in ('scan_vs_D_full', 'scan_vs_D_composed', 'scan_vs_A_full', 'scan_vs_A_composed', 'scan_vs_B_full_alone', 'scan_vs_C_composed_alone'))
    original_error_reproduced = all(comparisons['scan_vs_D_full'][k] == v for k, v in ORIGINAL_ERROR.items()) if not args.small else None
    outcome = {'completed_all_forward_contexts': True, 'ordinary_scan_capture_guards_exact': ordinary_guards,
        'same_original_source_input_and_output_hashes': original_operands_match,
        'original_D_error_statistics_reproduced': original_error_reproduced,
        'all_forward_contexts_exact_to_scan': all_contexts_match_scan,
        'forward_replay_boundary_valid': ordinary_guards and original_operands_match and all_contexts_match_scan,
        'full_trainer_native_attribution_tested': False,
        'no_new_parameter_or_cross_implementation_vjps': True,
        'D_retains_only_original_suffix_and_tail_input_vjps': True,
        'all_returned_buffers_blocked_read_and_finite': True,
        'note': 'Forward mismatch is retained as the object of this diagnostic; it is not numerical acceptance of training, a source fix, or a native comparison.'}
    prior.write_json(output / 'comparisons.json', comparisons)
    provenance = {'schema': 'native-splash-last-block-forward-context.v1', 'source_commit': SOURCE,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'frozen_original_script_sha256': FROZEN_BLOCK_SHA, 'frozen_capture_helper_sha256': CAPTURE_HELPER_SHA,
        'original_function_ast_proof': proof, 'source_file_sha256': frozen.source_hashes(),
        'execution_context_fidelity': 'D and source-scan ASTs match frozen858f, but total execution context is not identical. This driver blocks and host-reads source-scan buffers for statistics before D; the original scan-to-D dispatch had no such host inspection. Explicit lower/compile/HLO inspection, debug filename/line metadata and closed-over object identity can also differ. Exact source hashes and original-error statistics empirically test reproduction rather than assuming it.',
        'config': frozen.dataclasses.asdict(config), 'parameter_count': parameter_count,
        'parameter_sha256': parameter_hash, 'batch_sha256': batch_hash,
        'dtype': {'master': 'float32', 'compute': 'bfloat16'}, 'matmul_precision': 'default',
        'attention_backend': config.attention_backend.value, 'xla_flags': os.environ['XLA_FLAGS'],
        'jax': jax.__version__, 'process_count': jax.process_count(),
        'devices': [{'device': str(d), 'kind': d.device_kind, 'process_index': d.process_index,
            'coordinates': list(d.coords) if d.platform == 'tpu' else None,
            'core_on_chip': d.core_on_chip if d.platform == 'tpu' else None} for d in jax.devices()],
        'mesh': {'data': 2, 'model': 2}, 'axis_types': 'Auto',
        'context_definitions': {'A': 'Combined prefix+actual full+manual composed forward only; deliberately omits original suffix/tail input derivative branches.',
            'B': 'Actual unchanged named LlamaDecoderLayer.__call__ alone in a separate JIT.',
            'C': 'Manual prefix+tail alone in a separate JIT, same actual source submodules.',
            'D': 'AST-identical original capture_last including all6 original returned buffers and already-existing suffix value_and_grad/tail inputVJP; no new parameter/crossvariant VJPs.'},
        'source_equivalence_facts': ['LlamaDecoderLayer nonhybrid path equals residual+attention(inputnorm), then residual+MLP(postnorm).',
            'Configured no hybrid norm,bias,QK norm or dropout. maybe_rng_split(None) only returnsNone keys.',
            'haliax named_call is only jax.named_scope; scope metadata is an intended context difference, not a source math change.',
            'D and source scan ASTs include original argument order,decorators,body and returns. Filename/line debug metadata and explicit lower+compile invocation can still alter compilation context.'],
        'source_positions': statistics(positions.array), 'source_tokens': statistics(tokens), 'loss_weight': statistics(loss_weight),
        'original_reference_error': ORIGINAL_ERROR,
        'hypothesis': 'Determine whether the large named actual-block discrepancy persists standalone or only in joint graphs with duplicate Splash paths and/or existing source input derivative branches. CSE/scope/kernel-mask-sharding/lowering are hypotheses, not established causes.',
        'limits': {'hlo_per_file_bytes': HLO_FILE_LIMIT, 'hlo_total_bytes': HLO_TOTAL_LIMIT,
            'maximum_archive_entries': 128, 'maximum_archive_uncompressed_bytes': 64*1024*1024},
        'cpu_scope': 'Reference/vanilla harness only; not Splash execution' if args.small else None,
        'limitations': 'All paths are named legacy only; no native,optimizer,LR,tolerance,training/performance or convergence result. Intermediate outputs and additional graphs can affect compilation. A forward equality alone cannot validate full Trainer gradients.'}
    prior.write_json(output / 'provenance.json', provenance)
    files = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir())]
    outcome.update(hlo_bytes=capture.hlo_bytes, files=files, files_with_outcome=len(files)+1, bytes_before_outcome=sum(p['bytes'] for p in files))
    prior.write_json(output / 'case-outcomes.json', outcome)
    print('FORWARD_CONTEXT_OUTCOME ' + json.dumps({k:v for k,v in outcome.items() if k != 'files'}), flush=True)
    if not outcome['forward_replay_boundary_valid']:
        raise ValueError('Original/replay boundary failed in one or more contexts; evidence preserved without training/native attribution')


if __name__ == '__main__':
    main()
