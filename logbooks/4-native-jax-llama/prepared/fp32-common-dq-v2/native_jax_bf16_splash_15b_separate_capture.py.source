"""Last-block VJPs with actual-full source replay in a separate JIT graph."""
import argparse
import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import haliax as hax
from haliax.state_dict import from_torch_compatible_state_dict, flatten_modules_for_export, to_state_dict

sys.path.insert(0, str(Path(__file__).parent))
import native_jax_bf16_attention_replay as prior
from levanter.models.llama_native import NativeLlamaBlock, NativeLlamaSharding, NativeLlamaModel
from levanter.models.lm_model import LmExample

SOURCE = '36e2a9a97fab22b76dfa5a03964c6600ec5713ac'
CAPTURE_SHA = '365e8edbf32323e64528953e0122cdcc119da35ac93aaa2a8fa09f703d8f9615'
STAGES = ('full', 'prefix', 'tail', 'composed')
RESIDUAL = P('data', None, None)
STAGED_CONTROL_DESCRIPTION = ('Every variant tail receives the fixed captured named r and every variant prefix receives '
    'the fixed captured named ct_r. These are not each variant\'s own composed-chain intermediates/cotangents. '
    'Differences against full conflate fixed-boundary operand substitution, dtype and fusion/materialization; '
    'this is not a pure fusion test.')
LARGE_PARAMETER_COUNT = 1531037696
LARGE_PARAMETER_SHA = 'd79fb8c383f9af209fd9e3e32e00798c4aeb460d779a1444d9ea20ba9f453f9a'
LARGE_BATCH_SHA = 'd65acfa301bad7db399ff9df86cbbc150c649317dd357b3a26c53c9eaea9caa4'
SOURCE_FILES = ['lib/levanter/scripts/bench/bench_llama_native.py',
    'lib/levanter/src/levanter/models/llama.py',
    'lib/levanter/src/levanter/models/llama_native.py',
    'lib/levanter/src/levanter/layers/linear.py',
    'lib/levanter/src/levanter/layers/rms_norm.py',
    'lib/levanter/src/levanter/layers/attention.py',
    'lib/levanter/src/levanter/grug/attention/_core.py',
    'lib/levanter/src/levanter/kernels/pallas/splash_attention.py',
    'lib/haliax/src/haliax/nn/linear.py',
    'lib/haliax/src/haliax/nn/normalization.py',
    'lib/haliax/src/haliax/_src/state_dict.py']


FROZEN_BLOCK_PATH = Path(__file__).with_name('native_jax_bf16_splash_15b_block_replay.py')
FROZEN_BLOCK_SHA = '858f1885bf452b883d7e1cbaeec11c3beff3628bf6f674aab9789860853e5693'
CAPTURE_HLO_FILE_LIMIT = 4 * 1024 * 1024
CAPTURE_HLO_TOTAL_LIMIT = 24 * 1024 * 1024
ORIGINAL_OPERAND_HASHES = {
    'last_input': '610eaa4d595f6f112f77a507acf91b208930927bbe9c6d00582736d3ef5937ad',
    'last_output': 'ea3379fa53134daa136c29f4f3b7987a710a1e97a7e081da65a194c195e3fc9b',
    'post_attention_residual': '7dcfac5eb01887e14a42b8f885f65b07e7871e46755928003e65441b95b58b8b',
    'output_cotangent': '7596421ca57f20aadccca3d741aa70d076fc7657f2b029ca1dfa6e379a62059c',
    'tail_input_cotangent': '79d5859e723b097d4daa9173b932326a6dd0ff5466f3e47d1843cb88eedba793',
}


def frozen_definition_proof():
    original = ast.parse(FROZEN_BLOCK_PATH.read_text())
    current = ast.parse(Path(__file__).read_text())
    result = {}
    for name in ('configuration', 'abstract_shapes', 'canonical', 'weight_spec', 'scanned_capture', 'BlockReplay'):
        left = [n for n in ast.walk(original) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        right = [n for n in ast.walk(current) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        if len(left) != 1 or len(right) != 1:
            raise ValueError(f'Ambiguous frozen definition {name}')
        a, b = ast.dump(left[0], include_attributes=False), ast.dump(right[0], include_attributes=False)
        result[name] = {'identical_ast': a == b, 'ast_sha256': hashlib.sha256(a.encode()).hexdigest()}
        if a != b:
            raise ValueError(f'Frozen canonical/VJP boundary changed: {name}')
    old_capture = [n for n in ast.walk(original) if isinstance(n, ast.FunctionDef) and n.name == 'capture_last'][0]
    new_capture = [n for n in ast.walk(current) if isinstance(n, ast.FunctionDef) and n.name == 'capture_last'][0]
    old_capture.body = [n for n in old_capture.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'replayed_y' for t in n.targets))]
    old_return = old_capture.body[-1]
    old_return.value.elts = [item for item in old_return.value.elts if not (isinstance(item, ast.Name) and item.id == 'replayed_y')]
    a, b = ast.dump(old_capture, include_attributes=False), ast.dump(new_capture, include_attributes=False)
    result['capture_last'] = {'only_removed_duplicate_full_assignment_and_return': a == b, 'ast_sha256': hashlib.sha256(b.encode()).hexdigest()}
    if a != b:
        raise ValueError('Capture changed beyond removal of duplicate actual-full evaluation')
    return result


class SourceCapture:
    def __init__(self, output):
        self.output = output
        self.hlo_bytes = 0
        self.records = {}

    def run(self, label, fn, args, names, **kwargs):
        lowered = fn.lower(*args, **kwargs)
        compiled = lowered.compile()
        saved = {}
        for suffix, data in [('stablehlo', lowered.as_text().encode()), ('hlo', compiled.compiled.as_text().encode())]:
            keep = min(len(data), CAPTURE_HLO_FILE_LIMIT, CAPTURE_HLO_TOTAL_LIMIT - self.hlo_bytes)
            if keep <= 0:
                raise ValueError('Source capture HLO cap exceeded')
            self.hlo_bytes += keep
            filename = f'{label}.{suffix}.txt'
            (self.output / filename).write_bytes(data[:keep])
            saved[suffix] = {'file': filename, 'full_bytes': len(data), 'saved_bytes': keep,
                'sha256_full': hashlib.sha256(data).hexdigest(), 'sha256_saved': hashlib.sha256(data[:keep]).hexdigest(), 'truncated': keep < len(data)}
        result = compiled(*args, **kwargs)
        jax.block_until_ready(result)
        values = result if isinstance(result, tuple) else (result,)
        if len(values) != len(names):
            raise ValueError('Source result names must describe every returned buffer')
        records = {}
        for name, value in zip(names, values, strict=True):
            if isinstance(value, hax.NamedArray):
                value = value.array
            array = prior.host(value)
            float_array = array.astype(np.float32)
            if not np.isfinite(float_array).all():
                raise ValueError('Nonfinite source boundary')
            digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
            digest.update(array.tobytes())
            records[name] = {'sha256': digest.hexdigest(), 'shape': list(array.shape), 'dtype': str(array.dtype),
                'partition_spec': str(value.sharding.spec), 'sharding': str(value.sharding),
                'max_abs': float(np.max(np.abs(float_array))), 'rms': float(np.sqrt(np.mean(float_array * float_array))),
                'addressable_shards': [{'device': str(s.device), 'process': s.device.process_index,
                    'index': str(s.index), 'shape': list(s.data.shape)} for s in value.addressable_shards]}
        record = {'hlo': saved, 'result_order': list(names), 'results': records,
            'all_returned_buffers_blocked_read_and_finite': True, 'saved_hlo_before_execution_and_controls': True}
        prior.write_json(self.output / f'{label}.json', record)
        self.records[label] = record
        return values


def configuration(small):
    if small:
        # TPU Pallas Splash cannot execute on CPU. This validates the harness only.
        return prior.bench.BenchmarkConfig(vocab_size=64, hidden_dim=32, intermediate_dim=64,
            num_layers=2, num_heads=4, num_kv_heads=2, batch_size=4, seq_len=8,
            data_size=2, model_size=2, compute_dtype='bfloat16')
    return prior.bench.BenchmarkConfig(vocab_size=65536, hidden_dim=2048, intermediate_dim=5632,
        num_layers=28, num_heads=16, num_kv_heads=4, batch_size=8, seq_len=2048,
        data_size=2, model_size=2, compute_dtype='bfloat16',
        attention_backend=prior.bench.AttentionBackend.SPLASH)


def source_hashes():
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def abstract_shapes(config, output, base):
    sharding = NativeLlamaSharding(('data',), 'data', 'model')
    native_config = prior.bench._native_config(config)
    named_config = prior.bench._legacy_config(config)
    if config.attention_backend != prior.bench.AttentionBackend.SPLASH or native_config.attention_implementation != 'tpu_splash' or named_config.attn_backend != prior.bench.AttentionBackend.SPLASH:
        raise ValueError('Large abstract configuration must explicitly select both Splash implementations')
    with jax.set_mesh(base.meshes['native']):
        template = eqx.filter_eval_shape(NativeLlamaModel.init, native_config, key=jax.random.PRNGKey(config.seed), sharding=sharding)
        block = eqx.filter_eval_shape(lambda key: canonical(NativeLlamaBlock.init(native_config, sharding, key=key)), jax.random.PRNGKey(config.seed))
    parameter_count = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(template) if isinstance(leaf, jax.ShapeDtypeStruct))
    if parameter_count != LARGE_PARAMETER_COUNT:
        raise ValueError(f'Unexpected large model parameter count: {parameter_count}')
    with jax.set_mesh(base.meshes['named']), hax.axis_mapping(prior.bench._parameter_mapping()):
        named_block = eqx.filter_eval_shape(lambda key: canonical(named_config.build(hax.Axis('vocab', config.vocab_size), key=key).transformer.layers.unstacked()[-1]), jax.random.PRNGKey(config.seed))
    if set(named_block) != set(block) or any(named_block[k].shape != block[k].shape for k in block):
        raise ValueError('Large canonical named/native block shapes differ')
    record = {'source_commit': SOURCE, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_file_sha256': source_hashes(), 'config': dataclasses.asdict(config),
        'parameter_count': parameter_count, 'last_layer': config.num_layers - 1,
        'input_shape': [config.batch_size, config.seq_len, config.hidden_dim],
        'actual_head_dim': native_config.actual_head_dim,
        'named_attention_backend': named_config.attn_backend.value,
        'native_attention_implementation': native_config.attention_implementation,
        'canonical_block_shapes': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'storage_spec': str(weight_spec(k))} for k,v in block.items()},
        'compute_dtype': 'bfloat16', 'master_dtype': 'float32', 'fp32_reference': 'highest precision, same native tpu_splash attention',
        'no_model_parameters_allocated': True, 'no_forward_or_splash_kernel_compilation': True,
        'limitation': 'Abstract initialization/shape and explicit backend-selection proof only; not physical TPU or Splash execution.'}
    prior.write_json(output / 'abstract-shapes.json', record)
    print('LARGE_ABSTRACT_SHAPES ' + json.dumps({'parameter_count': parameter_count, 'last_layer': config.num_layers - 1, 'canonical_parameter_tensors': len(block)}), flush=True)


def canonical(block):
    return to_state_dict(flatten_modules_for_export(block))


def weight_spec(name):
    if 'layernorm' in name:
        return P('data')
    if name.endswith(('o_proj.weight', 'down_proj.weight')):
        return P('data', 'model')
    return P('model', 'data')


class BlockReplay(prior.Replay):
    def __init__(self, config, output, named_template, native_template):
        super().__init__(config, output)
        self.meshes['fp32'] = self.meshes['native']
        self.templates = {'named': named_template, 'native': native_template, 'fp32': native_template}
        self.keys = tuple(sorted(canonical(named_template)))
        if set(self.keys) != set(canonical(native_template)):
            raise ValueError('Actual named/native block parameter keys differ')
        self.positions_host = self.positions

    def functions(self, variant, stage):
        def execute(x, *weights):
            # Use the existing canonical deserializer, not a second parameter loader.
            with hax.axis_mapping(prior.bench._parameter_mapping()):
                block = from_torch_compatible_state_dict(self.templates[variant], dict(zip(self.keys, weights, strict=True)))
            dtype = jnp.float32 if variant == 'fp32' else jnp.bfloat16
            block = jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, block)
            positions = jax.device_put(self.positions_host, NamedSharding(self.meshes[variant], P('data', None)))
            if variant == 'named':
                named_positions = hax.named(positions, (self.batch, self.pos))
                value = hax.named(x, (self.batch, self.pos, self.embed))
                def prefix(v):
                    return v + block.self_attn(block.input_layernorm(v), prior.NamedMask.causal(), pos_ids=named_positions)
                def tail(v):
                    return v + block.mlp(block.post_attention_layernorm(v))
                if stage == 'full':
                    result = block(value, prior.NamedMask.causal(), pos_ids=named_positions)
                elif stage == 'prefix':
                    result = prefix(value)
                elif stage == 'tail':
                    result = tail(value)
                else:
                    result = tail(prefix(value))
                return result.rearrange((self.batch, self.pos, self.embed)).array
            def prefix(v):
                return v + block.self_attn(block.input_layernorm(v), prior.AttentionMask.causal(), positions)
            def tail(v):
                return v + block.mlp(block.post_attention_layernorm(v))
            if stage == 'full':
                return block(x, prior.AttentionMask.causal(), positions)
            if stage == 'prefix':
                return prefix(x)
            if stage == 'tail':
                return tail(x)
            return tail(prefix(x))
        return execute

    def run(self, variant, stage, x, weights, ct):
        mesh = self.meshes[variant]
        specs = (RESIDUAL, *(weight_spec(k) for k in self.keys))
        raw = (prior.host(x), *(prior.host(weights[k]) for k in self.keys))
        raw_ct = prior.host(ct)
        if raw[0].dtype != np.dtype(jnp.bfloat16) or raw_ct.dtype != np.dtype(jnp.bfloat16):
            raise ValueError('Captured block input/cotangent must be BF16')
        if any(a.dtype != np.dtype(np.float32) for a in raw[1:]):
            raise ValueError('Master weights must be FP32')
        with jax.set_mesh(mesh), hax.axis_mapping(prior.bench._compute_mapping()), jax.default_matmul_precision('highest' if variant == 'fp32' else 'default'):
            args = tuple(jax.device_put(a.astype(np.float32) if variant == 'fp32' else a, NamedSharding(mesh, spec)) for a, spec in zip(raw, specs, strict=True))
            cotangent = jax.device_put(raw_ct.astype(np.float32) if variant == 'fp32' else raw_ct, NamedSharding(mesh, RESIDUAL))
            if variant != 'fp32' and not all(np.array_equal(a, prior.host(b)) for a, b in zip((*raw, raw_ct), (*args, cotangent), strict=True)):
                raise ValueError('Common operand import changed values')
            fn = self.functions(variant, stage)
            count = len(args)
            def execute(*packed):
                output, backward = jax.vjp(fn, *packed[:count])
                return (output, *backward(packed[count]))
            output_shardings = tuple(NamedSharding(mesh, s) for s in (RESIDUAL, *specs))
            execute = jax.jit(execute, out_shardings=output_shardings)
            lowered = execute.lower(*args, cotangent)
            stable = lowered.as_text().encode()
            compiled = lowered.compile()
            result = compiled(*args, cotangent)
            jax.block_until_ready(result)
            if any(a.dtype != jnp.float32 for a in result[2:]):
                raise ValueError('Master-parameter VJPs must remain FP32')
            saved = {}
            for suffix, data in [('hlo', compiled.as_text().encode()), ('stablehlo', stable)]:
                keep = min(len(data), prior.HLO_FILE_LIMIT, prior.HLO_TOTAL_LIMIT - self.hlo_bytes)
                if keep <= 0:
                    raise ValueError('HLO cap exceeded')
                self.hlo_bytes += keep
                path = f'last-{variant}-{stage}.{suffix}.txt'
                (self.output / path).write_bytes(data[:keep])
                saved[suffix] = {'file': path, 'full_bytes': len(data), 'saved_bytes': keep, 'truncated': keep < len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            record = {'matmul_precision': str(jax.config.jax_default_matmul_precision),
                'arguments': [prior.statistics(a) | {'partition_spec': str(a.sharding.spec), 'sharding': str(a.sharding)} for a in args],
                'cotangent': prior.statistics(cotangent) | {'partition_spec': str(cotangent.sharding.spec), 'sharding': str(cotangent.sharding)},
                'results': [prior.statistics(a) | {'partition_spec': str(a.sharding.spec), 'sharding': str(a.sharding)} for a in result],
                'result_order': ['forward', 'input_vjp', *self.keys], 'hlo': saved,
                'stablehlo_dot_lines': [line.strip()[:1500] for line in stable.decode().splitlines() if 'stablehlo.dot_general' in line][:40]}
            return tuple(prior.host(a) for a in result), record

    def compare(self, stage, x, weights, ct):
        values, records = {}, {}
        for variant in ('named', 'native', 'fp32'):
            values[variant], records[variant] = self.run(variant, stage, x, weights, ct)
        for a, b in zip(records['named']['arguments'], records['native']['arguments'], strict=True):
            if any(a[k] != b[k] for k in ('sha256', 'shape', 'dtype', 'partition_spec')):
                raise ValueError('Common named/native master/input boundary differs')
        a, b = records['named']['cotangent'], records['native']['cotangent']
        if any(a[k] != b[k] for k in ('sha256', 'shape', 'dtype', 'partition_spec')):
            raise ValueError('Common named/native cotangent boundary differs')
        records['common_bf16_operands_fp32_masters_and_physical_specs_exact'] = True
        records['comparisons'] = {name: [prior.difference(a, b) for a, b in zip(values[left], values[right], strict=True)] for name, left, right in [('named_vs_native', 'named', 'native'), ('named_vs_fp32', 'named', 'fp32'), ('native_vs_fp32', 'native', 'fp32')]}
        prior.write_json(self.output / f'last-{stage}.json', records)
        print('BLOCK_REPLAY ' + json.dumps({'stage': stage, 'named_vs_native': records['comparisons']['named_vs_native']}), flush=True)
        return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--abstract-only', action='store_true')
    parser.add_argument('--expected-backend', choices=('cpu', 'tpu'), required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get('IRIS_OUTPUT_DIR', '/tmp/native-bf16-block-replay')))
    args = parser.parse_args()
    if hashlib.sha256(FROZEN_BLOCK_PATH.read_bytes()).hexdigest() != FROZEN_BLOCK_SHA:
        raise ValueError('Frozen original diagnostic changed')
    definition_proof = frozen_definition_proof()
    if hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest() != CAPTURE_SHA:
        raise ValueError('Frozen capture helper changed')
    if os.environ.get('XLA_FLAGS') != '--xla_allow_excess_precision=false' or os.environ.get('SOURCE_COMMIT', SOURCE) != SOURCE:
        raise ValueError('Expected exact source/XLA flags')
    if jax.process_count() != 1 or jax.device_count() != 4 or jax.local_device_count() != 4 or jax.default_backend() != args.expected_backend:
        raise ValueError('Expected one process/four devices/backend')
    if not args.small and not args.abstract_only and (args.expected_backend != 'tpu' or any('v5' not in d.device_kind.lower() or 'lite' in d.device_kind.lower() for d in jax.devices())):
        raise ValueError('Full replay requires v5p')
    jax.config.update('jax_default_matmul_precision', 'default')
    config = configuration(args.small)
    if not args.small and (prior.bench._legacy_config(config).attn_backend != prior.bench.AttentionBackend.SPLASH or prior.bench._native_config(config).attention_implementation != 'tpu_splash'):
        raise ValueError('Actual large named/native blocks must both select Splash')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use fresh empty output directory')
    base = prior.Replay(config, output)
    source_capture = SourceCapture(output)
    if args.abstract_only:
        if args.small:
            raise ValueError('Use abstract-only for the actual large Splash configuration')
        abstract_shapes(config, output, base)
        return
    named_mesh, native_mesh = base.meshes['named'], base.meshes['native']
    tokens_host = np.random.default_rng(config.seed + 2).integers(0, config.vocab_size, (config.batch_size, config.seq_len), dtype=np.int32)
    loss_weight_host = np.ones_like(tokens_host, dtype=np.float32)
    loss_weight_host[:, -1] = 0
    with jax.set_mesh(named_mesh), hax.axis_mapping(prior.bench._parameter_mapping()):
        master = hax.named_jit(prior.bench._legacy_config(config).build, axis_resources=prior.bench._parameter_mapping())(hax.Axis('vocab', config.vocab_size), key=jax.random.PRNGKey(config.seed))
        parameter_count = sum(int(np.prod(a.shape)) for a in jax.tree.leaves(master) if eqx.is_inexact_array(a))
        if not args.small and parameter_count != LARGE_PARAMETER_COUNT:
            raise ValueError('Unexpected actual large source model parameter count')
        parameter_hash = prior.bench._tensor_digest(prior.bench._canonical_parameters(master))
        named_template = master.transformer.layers.unstacked()[-1]
        weights = canonical(named_template)
        if any(a.dtype != jnp.float32 for a in weights.values()):
            raise ValueError('Source masters not FP32')
        model = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if eqx.is_inexact_array(a) else a, master)
        tokens = jax.device_put(tokens_host, NamedSharding(named_mesh, P('data', None)))
        loss_weight = jax.device_put(loss_weight_host, NamedSharding(named_mesh, P('data', None)))
        named_tokens = hax.named(tokens, (base.batch, base.pos))
        positions = hax.named(jax.device_put(base.positions, NamedSharding(named_mesh, P('data', None))), (base.batch, base.pos))
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
            def tail(residual):
                h = hax.named(residual, axes)
                return (h + block.mlp(block.post_attention_layernorm(h))).rearrange(axes).array
            def suffix(hidden):
                h = m.transformer.norm(hax.named(hidden, axes)).rearrange(axes).array
                return prior.bench._shared_loss(h, m.get_lm_head().rearrange(('vocab', 'embed')).array, tokens, loss_weight)
            loss, ct_y = jax.value_and_grad(suffix)(y)
            staged_y, pb_tail = jax.vjp(tail, r.rearrange(axes).array)
            ct_r = pb_tail(ct_y)[0]
            return r.rearrange(axes).array, staged_y, loss, ct_y, ct_r
        @eqx.filter_jit
        def isolated_full(m, x):
            block = m.transformer.layers.unstacked()[-1]
            axes = (base.batch, base.pos, base.embed)
            return block(hax.named(x, axes), prior.NamedMask.causal(), pos_ids=positions).rearrange(axes).array
        with hax.axis_mapping(prior.bench._compute_mapping()):
            x, y, hidden = source_capture.run('source-scan', scanned_capture, (model, named_tokens), ('last_input', 'last_output', 'normalized_output'))
            # The duplicate actual-full call is absent from this capture graph.
            r, staged_y, loss, ct_y, ct_r = source_capture.run('capture-without-duplicate-full', capture_last, (model, x, y), ('residual', 'composed', 'suffix_loss', 'output_cotangent', 'tail_input_cotangent'))
            replayed_y, = source_capture.run('source-full-separate', isolated_full, (model, x), ('full',))
            normal_named_hidden, = source_capture.run('normal-hidden', eqx.filter_jit(lambda m, t: m.activations(t, prior.NamedMask.causal(), pos_ids=positions)), (model, named_tokens), ('normalized_output',))
            normal_hidden = normal_named_hidden.rearrange((base.batch, base.pos, base.embed)).array
            batch = LmExample(named_tokens, hax.named(loss_weight, (base.batch, base.pos)), prior.NamedMask.causal())
            normal_loss, = source_capture.run('normal-loss', eqx.filter_jit(prior.bench._legacy_loss), (model, batch), ('loss',), key=None)
    with jax.set_mesh(native_mesh):
        native_template = eqx.filter_jit(NativeLlamaBlock.init)(prior.bench._native_config(config), NativeLlamaSharding(('data',), 'data', 'model'), key=jax.random.PRNGKey(config.seed + 3))
    replay = BlockReplay(config, output, named_template, native_template)
    with jax.set_mesh(named_mesh):
        imported_named = eqx.filter_jit(lambda block, state: from_torch_compatible_state_dict(block, state))(named_template, weights)
        named_roundtrip = canonical(imported_named)
    with jax.set_mesh(native_mesh):
        imported_native = eqx.filter_jit(lambda block, state: from_torch_compatible_state_dict(block, state))(native_template, {k: jax.device_put(prior.host(v), NamedSharding(native_mesh, weight_spec(k))) for k, v in weights.items()})
        native_roundtrip = canonical(imported_native)
    roundtrip = {k: {'named': prior.difference(weights[k], named_roundtrip[k]), 'native': prior.difference(weights[k], native_roundtrip[k])} for k in replay.keys}
    if not all(pair['named']['exact'] and pair['native']['exact'] for pair in roundtrip.values()):
        raise ValueError('Canonical existing-loader master import changed values')
    provenance = {'schema': 'native-bf16-common-splash-15b-last-block-separate-capture.v1', 'source_commit': SOURCE, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'frozen_capture_sha256': CAPTURE_SHA, 'config': dataclasses.asdict(config), 'attention_selection': {'named': prior.bench._legacy_config(config).attn_backend.value, 'native': prior.bench._native_config(config).attention_implementation, 'fp32_reference': prior.bench._native_config(config).attention_implementation}, 'small_cpu_scope': 'Reference-attention harness proof only; no Splash kernel execution' if args.small else None, 'layer': config.num_layers - 1, 'parameter_sha256': parameter_hash, 'batch_sha256': prior.bench._tensor_digest({'tokens': tokens_host, 'loss_weight': loss_weight_host}), 'source_file_sha256': source_hashes(), 'devices': [{'id': d.id, 'kind': d.device_kind, 'process_index': d.process_index, 'description': str(d), 'coordinates': list(d.coords) if d.platform == 'tpu' else None, 'core_on_chip': d.core_on_chip if d.platform == 'tpu' else None} for d in jax.devices()], 'process_count': jax.process_count(), 'global_device_count': jax.device_count(), 'local_device_count': jax.local_device_count(), 'jax': jax.__version__, 'xla_flags': os.environ['XLA_FLAGS'], 'variant_matmul_precision': {'named': 'default', 'native': 'default', 'fp32': 'highest'}, 'ordinary_scan_loss': float(normal_loss), 'source_suffix_loss': float(loss), 'ordinary_scan_vs_captured_hidden': prior.difference(normal_hidden, hidden), 'suffix_vs_scan_loss': prior.difference(normal_loss, loss), 'source_scan_vs_replayed_last_block': prior.difference(y, replayed_y), 'source_full_vs_staged_forward': prior.difference(y, staged_y), 'canonical_master_roundtrip': roundtrip, 'source_operands': {name: prior.statistics(value) | {'partition_spec': str(value.sharding.spec)} for name, value in [('last_input', x), ('post_attention_residual', r), ('last_output', y), ('output_cotangent', ct_y), ('tail_input_cotangent', ct_r)]}, 'control_definitions': {'full': 'Actual unchanged decoder __call__', 'composed': 'Same actual source modules called as prefix then tail in a single JIT; no intermediate output materialization', 'tail': 'Common source r plus actual MLP(post_attention_layernorm(r)); shared source suffix cotangent', 'prefix': 'Common source x plus actual attention(input_layernorm(x)); shared source tail input cotangent', 'staged_vs_full': STAGED_CONTROL_DESCRIPTION, 'fp32_controls': 'FP32 highest native modules with the same selected attention backend and captured BF16 operands/cotangents cast to FP32. Large named/native/FP32 all use Splash; CPU-small all use vanilla/reference. Dtype, matmul precision and Auto/Explicit fusion contexts still differ. The independent tail uses BF16 source r, so staged/full equivalence is not a reference assumption'}, 'limits': {'hlo_file_bytes': prior.HLO_FILE_LIMIT, 'total_hlo_bytes': prior.HLO_TOTAL_LIMIT, 'expected_script_files': 31, 'wrapper_entry_limit': 128, 'wrapper_total_bytes': 64 * 1024 * 1024}, 'limitations': 'Source suffix cotangent comes from final norm and shared CE on the named legacy last output; it is not an independently native incoming cotangent or a hook-free full Trainer gradient. Capturing intermediates, canonical import constraints, VJP outputs and standalone execution can change fusion. Exact source capture-versus-normal hidden/loss and replayed-last-block output checks are mandatory before attribution; these do not establish identity of every full Trainer hidden cotangent. Scan capture retains only the last input in carry; it does not unroll28layers or save intermediate tensors. No optimizer, training threshold, model math or source edits; no tensor/weight/checkpoint artifacts.'}
    provenance['frozen_definition_proof'] = definition_proof
    provenance['graph_context_change'] = {
        'frozen_original_script_sha256': FROZEN_BLOCK_SHA,
        'changed_capture': 'Remove only actual-full block assignment and return from capture_last. Prefix, tail input VJP and suffix value_and_grad remain. Obtain actual-full source output in separate isolated_full JIT.',
        'prior_fault_preserved': 'Frozen original and combined-forward graph full branch differs from scan by relativeL2 0.16780116533211895, maxabs4.19921875; separate actual-full and manual-composed forwards match scan exactly on physical v5p. This diagnostic changes compilation context, not model semantics or acceptance guards.',
        'capture_vs_normal_limit': 'Host materialization and explicit lower/compile/HLO before controls perturb total compilation/execution context. Exact new source/ordinary forward/loss/hash guards are required; this does not establish full-Trainer/native backward equivalence.',
        'native_full_trainer_attribution_tested': False}
    provenance['source_capture_hlo_bytes'] = source_capture.hlo_bytes
    provenance['source_operand_hashes_vs_original'] = {name: {'expected': digest, 'actual': provenance['source_operands'][name]['sha256'], 'exact': provenance['source_operands'][name]['sha256'] == digest} for name, digest in ORIGINAL_OPERAND_HASHES.items()}
    provenance['source_operand_hashes_exact'] = args.small or all(record['exact'] for record in provenance['source_operand_hashes_vs_original'].values())
    provenance['limits'] = {'source_hlo_file_bytes': CAPTURE_HLO_FILE_LIMIT, 'source_hlo_total_bytes': CAPTURE_HLO_TOTAL_LIMIT,
        'vjp_hlo_file_bytes': prior.HLO_FILE_LIMIT, 'vjp_hlo_total_bytes': prior.HLO_TOTAL_LIMIT,
        'combined_hlo_total_bytes': CAPTURE_HLO_TOTAL_LIMIT + prior.HLO_TOTAL_LIMIT,
        'expected_script_files': 46, 'wrapper_entry_limit': 128, 'wrapper_total_bytes': 64 * 1024 * 1024}
    if not args.small and (parameter_hash != LARGE_PARAMETER_SHA or provenance['batch_sha256'] != LARGE_BATCH_SHA):
        prior.write_json(output / 'provenance.json', provenance)
        raise ValueError('Large source weights/batch differ from the measured1.53B Splash case')
    capture_checks = ('ordinary_scan_vs_captured_hidden', 'suffix_vs_scan_loss', 'source_scan_vs_replayed_last_block', 'source_full_vs_staged_forward')
    provenance['capture_controls_exact'] = all(provenance[key]['exact'] for key in capture_checks)
    provenance['parameter_count'] = parameter_count
    provenance['source_cotangents_nonzero'] = all(provenance['source_operands'][key]['rms'] > 0 for key in ('output_cotangent', 'tail_input_cotangent'))
    prior.write_json(output / 'provenance.json', provenance)
    if not provenance['capture_controls_exact'] or not provenance['source_cotangents_nonzero'] or not provenance['source_operand_hashes_exact']:
        prior.write_json(output / 'case-outcomes.json', {'completed': False, 'attribution_valid': False, 'reason': 'Source capture/ordinary scan controls not exact; no cross-implementation VJP attribution claimed'})
        raise ValueError('Source capture controls are not exact; preserve evidence and stop')
    values = {stage: replay.compare(stage, r if stage == 'tail' else x, weights, ct_r if stage == 'prefix' else ct_y) for stage in STAGES}
    controls = {'source_output_vs_named_full': prior.difference(y, values['full']['named'][0]), 'source_residual_vs_named_prefix': prior.difference(r, values['prefix']['named'][0]), 'source_tail_cotangent_vs_named_tail': prior.difference(ct_r, values['tail']['named'][1]), 'full_vs_composed': {}, 'staged_vs_full': {}, 'staged_vs_full_definition': STAGED_CONTROL_DESCRIPTION}
    for variant in ('named', 'native', 'fp32'):
        controls['full_vs_composed'][variant] = [prior.difference(a, b) for a, b in zip(values['full'][variant], values['composed'][variant], strict=True)]
        # Prefix/tail use fixed named r/ct_r, not this variant's own chain operands.
        stitched = (values['tail'][variant][0], values['prefix'][variant][1], *(a + b for a, b in zip(values['prefix'][variant][2:], values['tail'][variant][2:], strict=True)))
        controls['staged_vs_full'][variant] = [prior.difference(a, b) for a, b in zip(values['full'][variant], stitched, strict=True)]
    replay_control_keys = ('source_output_vs_named_full', 'source_residual_vs_named_prefix', 'source_tail_cotangent_vs_named_tail')
    controls['source_replay_boundaries_exact'] = all(controls[key]['exact'] for key in replay_control_keys)
    controls['single_jit_full_vs_composed_exact'] = {variant: all(value['exact'] for value in results) for variant, results in controls['full_vs_composed'].items()}
    controls['attribution_valid'] = controls['source_replay_boundaries_exact'] and all(controls['single_jit_full_vs_composed_exact'].values())
    prior.write_json(output / 'controls.json', controls)
    files = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir())]
    prior.write_json(output / 'case-outcomes.json', {'completed': True, 'attribution_valid': controls['attribution_valid'], 'stages': list(STAGES), 'all_values_finite': True, 'hlo_bytes': replay.hlo_bytes + source_capture.hlo_bytes, 'source_capture_hlo_bytes': source_capture.hlo_bytes, 'vjp_hlo_bytes': replay.hlo_bytes, 'files': files, 'files_with_manifest': len(files) + 1, 'bytes_before_manifest': sum(f['bytes'] for f in files)})
    print('BLOCK_REPLAY_DONE ' + json.dumps({'files_with_manifest': len(files) + 1, 'hlo_bytes': replay.hlo_bytes + source_capture.hlo_bytes, 'attribution_valid': controls['attribution_valid']}), flush=True)
    if not controls['attribution_valid']:
        raise ValueError('Source/replay boundary or single-JIT full/composed controls differ; no full-Trainer attribution')


if __name__ == '__main__':
    main()
