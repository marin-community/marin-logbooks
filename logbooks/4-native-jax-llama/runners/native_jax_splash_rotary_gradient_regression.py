"""Public RoPE/Splash gradient-parity candidate; not a proven regression test."""
import argparse
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.attention import AttentionMask, attention
from levanter.layers.attention import AttentionBackend, AttentionMask as NamedAttentionMask, dot_product_attention
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.layers.rotary_native import RotaryConfig, apply_rotary

BASE = '36e2a9a97fab22b76dfa5a03964c6600ec5713ac'
CANDIDATE = '8aa8674d10894728ed0b455ae14e14f46457cf9f'
CORE_SHA = {BASE: '07c8518380ee8a62cec006adf85458979be590630051e7c62dc9f7cb1285c3ec', CANDIDATE: 'e9bb1639396b5060fc6f0ebc013a3b89daf42485c88ea546b346198e9226c3d9'}
BATCH, SEQUENCE, QUERY_HEADS, KV_HEADS = 4, 512, 16, 4
ATOL, RTOL = 1e-5, 1e-4
MAPPING = {'batch': 'data', 'kv_head': 'model'}


def functions(head_dim):
    batch, pos, key_pos, kv, group, dim = (
        hax.Axis(name, size) for name, size in [('batch', BATCH), ('position', SEQUENCE), ('key_position', SEQUENCE), ('kv_head', KV_HEADS), ('q_heads_per_group', QUERY_HEADS // KV_HEADS), ('head_size', head_dim)]
    )
    rope = DefaultRotaryEmbeddingsConfig(theta=10000.0, factor=1.0).build(dim)

    def named(q, k, v, positions):
        q, k, v = (x.astype(jnp.bfloat16) for x in (q, k, v))
        p = hax.named(positions, (batch, pos))
        query = rope(hax.named(q.reshape(BATCH, SEQUENCE, KV_HEADS, group.size, head_dim), (batch, pos, kv, group, dim)), p)
        key = rope(hax.named(k, (batch, pos, kv, dim)), p).rename({pos.name: key_pos.name})
        value = hax.named(v, (batch, key_pos, kv, dim))
        output = dot_product_attention(
            pos, key_pos, dim,
            query.rearrange((batch, kv, group, pos, dim)),
            key.rearrange((batch, kv, key_pos, dim)),
            value.rearrange((batch, kv, key_pos, dim)),
            mask=NamedAttentionMask.causal(), attn_backend=AttentionBackend.SPLASH,
            attention_dtype=jnp.bfloat16, inference=True,
        )
        return output.rearrange((batch, pos, kv, group, dim)).array.reshape(q.shape)

    def native(q, k, v, positions):
        q, k, v = (x.astype(jnp.bfloat16) for x in (q, k, v))
        q = apply_rotary(q, positions, RotaryConfig())
        k = apply_rotary(k, positions, RotaryConfig())
        return attention(q, k, v, AttentionMask.causal(), implementation='tpu_splash')

    def objective(forward, q, k, v, positions, cotangent):
        output = forward(q, k, v, positions)
        return jnp.sum(output.astype(jnp.float32) * cotangent), output

    return {name: jax.value_and_grad(functools.partial(objective, fn), argnums=(0, 1, 2), has_aux=True) for name, fn in [('named', named), ('native', native)]}


def summary(value):
    array = np.array(value, copy=True)
    digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    digest.update(array.tobytes())
    fp32 = array.astype(np.float32)
    if not np.isfinite(fp32).all():
        raise ValueError('Nonfinite public API value/gradient')
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'partition_spec': str(value.sharding.spec), 'sha256': digest.hexdigest(), 'rms': float(np.sqrt(np.mean(fp32 * fp32)))}


def compare(left, right):
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    delta = a - b
    return {'allclose': bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)), 'exact': bool(np.array_equal(a, b)), 'max_abs': float(np.max(np.abs(delta))), 'mean_abs': float(np.mean(np.abs(delta))), 'relative_l2_named_denominator': float(np.linalg.norm(delta) / max(float(np.linalg.norm(a)), 1e-30)), 'different_elements': int(np.count_nonzero(delta)), 'elements': int(delta.size)}


def run_case(head_dim, meshes, abstract):
    forward = functions(head_dim)
    shapes = [(BATCH, SEQUENCE, QUERY_HEADS, head_dim), (BATCH, SEQUENCE, KV_HEADS, head_dim), (BATCH, SEQUENCE, KV_HEADS, head_dim), (BATCH, SEQUENCE), (BATCH, SEQUENCE, QUERY_HEADS, head_dim)]
    specs = [P('data', None, 'model', None)] * 3 + [P('data', None), P('data', None, 'model', None)]
    def compiled(variant):
        mesh = meshes[variant]
        residual = NamedSharding(mesh, specs[0])
        return jax.jit(forward[variant], out_shardings=((NamedSharding(mesh, P()), residual), (residual, residual, residual)))
    record = {'head_dim': head_dim, 'shape_q': list(shapes[0]), 'mask': 'causal', 'named_backend': 'splash', 'native_backend': 'tpu_splash', 'master_dtype': 'float32', 'compute_dtype': 'bfloat16', 'cotangent': 'Deterministic standard-normal F32 normalized to unit RMS, cast to BF16 by output-cast transpose; no mean-loss normalization or selected amplification.'}
    if abstract:
        results = {}
        for variant in ['named', 'native']:
            mesh = meshes[variant]
            args = [jax.ShapeDtypeStruct(shape, jnp.int32 if i == 3 else jnp.float32, sharding=NamedSharding(mesh, spec)) for i, (shape, spec) in enumerate(zip(shapes, specs, strict=True))]
            with jax.set_mesh(mesh), hax.axis_mapping(MAPPING):
                result = jax.eval_shape(compiled(variant), *args)
            results[variant] = [{'shape': list(x.shape), 'dtype': str(x.dtype), 'sharding': str(x.sharding)} for x in jax.tree.leaves(result)]
        record['abstract_outputs'] = results
        record['no_numerical_execution_or_splash_compilation'] = True
        return record
    rng = np.random.default_rng(173)
    arrays = [rng.standard_normal(shape).astype(np.float32) for shape in shapes[:3]]
    arrays.append(np.broadcast_to(np.arange(SEQUENCE, dtype=np.int32), shapes[3]).copy())
    cotangent = rng.standard_normal(shapes[4]).astype(np.float32)
    cotangent /= np.sqrt(np.mean(cotangent.astype(np.float64) ** 2))
    arrays.append(cotangent)
    values = {}
    for variant in ['named', 'native']:
        mesh = meshes[variant]
        args = [jax.device_put(value, NamedSharding(mesh, spec)) for value, spec in zip(arrays, specs, strict=True)]
        with jax.set_mesh(mesh), hax.axis_mapping(MAPPING):
            result = compiled(variant)(*args)
            jax.block_until_ready(result)
        values[variant] = [result[0][0], result[0][1], *result[1]]
        record[variant] = dict(zip(['loss', 'forward', 'query_master_vjp', 'key_master_vjp', 'value_master_vjp'], (summary(x) for x in values[variant]), strict=True))
    record['comparisons'] = dict(zip(['loss', 'forward', 'query_master_vjp', 'key_master_vjp', 'value_master_vjp'], (compare(a, b) for a, b in zip(values['named'], values['native'], strict=True)), strict=True))
    record['passed'] = all(v['allclose'] for v in record['comparisons'].values())
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abstract-only', action='store_true')
    parser.add_argument('--head64-control', action='store_true')
    parser.add_argument('--source-pin', choices=[BASE, CANDIDATE], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    core = Path(inspect.getsourcefile(attention))
    if hashlib.sha256(core.read_bytes()).hexdigest() != CORE_SHA[args.source_pin]:
        raise ValueError('Actual attention source differs from requested frozen pin')
    if os.environ.get('XLA_FLAGS') != '--xla_allow_excess_precision=false':
        raise ValueError('Expected unchanged shared XLA precision flags')
    if jax.device_count() != 4 or jax.process_count() != 1:
        raise ValueError('Requires exactly four actual devices/one process for 2x2 physical layout')
    if not args.abstract_only and jax.default_backend() != 'tpu':
        raise ValueError('Numerical Splash execution requires actual TPU; CPU can only eval_shape')
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory')
    args.output_dir.mkdir(parents=True)
    jax.config.update('jax_default_matmul_precision', 'default')
    devices = np.asarray(jax.devices()).reshape(2, 2)
    meshes = {name: Mesh(devices, ('data', 'model'), axis_types=(axis, axis)) for name, axis in [('named', AxisType.Auto), ('native', AxisType.Explicit)]}
    records = [run_case(dim, meshes, args.abstract_only) for dim in ([128, 64] if args.head64_control else [128])]
    report = {'status': 'abstract_only' if args.abstract_only else 'executed', 'source_pin': args.source_pin, 'core_sha256': CORE_SHA[args.source_pin], 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'backend': jax.default_backend(), 'devices': [str(x) for x in jax.devices()], 'atol': ATOL, 'rtol': RTOL, 'cases': records, 'scope': 'Proposed public-API regression only. No claim36e fails/8aa passes until both run on actual TPU. Abstract proof never executesSplash. Unit-RMS cotangent defines a per-unit derivative test; does not change production gate or reproduce mean-CE normalization.', 'full_trainer_identity_tested': False}
    (args.output_dir / 'proof.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'case_count': len(records), 'passed': None if args.abstract_only else all(x['passed'] for x in records)}), flush=True)
    if not args.abstract_only and not all(x['passed'] for x in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
