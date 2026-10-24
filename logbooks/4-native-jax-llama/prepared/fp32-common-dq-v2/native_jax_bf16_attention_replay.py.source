"""Common-operand attention diagnostic; no training, model patch, or tensor artifacts."""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, auto_axes
import numpy as np
import haliax as hax

sys.path.insert(0, str(Path('lib/levanter/scripts/bench').resolve()))
import bench_llama_native as bench
from levanter.grug.attention import AttentionMask, reference_attention
from levanter.layers.attention import AttentionMask as NamedMask, simple_attention_with_dropout
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.layers.rotary_native import RotaryConfig, apply_rotary
from levanter.models.lm_model import LmExample

SOURCE = '0acafd210b88c22011a5808b9c4b68d8eb088f25'
HLO_FILE_LIMIT = 1024 * 1024
HLO_TOTAL_LIMIT = 32 * 1024 * 1024


def host(value):
    return np.array(jax.device_get(value), copy=True)


def digest(value):
    array = host(value)
    result = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    result.update(array.view(np.uint8).tobytes())
    return result.hexdigest()


def statistics(value):
    x = host(value).astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError('Nonfinite diagnostic operand/result')
    return {'shape': list(x.shape), 'dtype': str(value.dtype), 'sha256': digest(value),
            'max_abs': float(np.max(np.abs(x))), 'rms': float(np.sqrt(np.mean(x*x)))}


def difference(a, b):
    x, y = host(a).astype(np.float64), host(b).astype(np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Nonfinite comparison')
    delta = x-y
    ix = np.unravel_index(int(np.argmax(np.abs(delta))), x.shape)
    return {'exact': bool(np.array_equal(x,y)), 'max_abs': float(np.max(np.abs(delta))),
            'relative_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(x),1e-30)),
            'different_elements': int(np.sum(x != y)), 'elements': int(x.size),
            'worst_index': [int(i) for i in ix], 'a_at_worst': float(x[ix]), 'b_at_worst': float(y[ix])}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


class Replay:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.hlo_bytes = 0
        devices = np.asarray(jax.devices()).reshape(2,2)
        self.meshes = {name: Mesh(devices, ('data','model'), axis_types=(kind,kind))
                       for name,kind in [('named',AxisType.Auto),('native',AxisType.Explicit),('fp32',AxisType.Auto)]}
        self.batch = hax.Axis('batch',config.batch_size)
        self.pos = hax.Axis('position',config.seq_len)
        self.kpos = self.pos.alias('key_position')
        self.kv = hax.Axis('kv_head',config.num_kv_heads)
        self.group = hax.Axis('q_heads_per_group',config.num_heads//config.num_kv_heads)
        self.dim = hax.Axis('head_size',config.hidden_dim//config.num_heads)
        self.embed = hax.Axis('embed',config.hidden_dim)
        self.spec = P('data',None,'model',None)
        self.score_spec = P('data','model',None,None,None)
        self.positions = np.broadcast_to(np.arange(config.seq_len,dtype=np.int32),(config.batch_size,config.seq_len)).copy()

    def named_q(self, q):
        return hax.named(q.reshape(self.config.batch_size,self.config.seq_len,self.kv.size,self.group.size,self.dim.size),
                         (self.batch,self.pos,self.kv,self.group,self.dim)).rearrange((self.batch,self.kv,self.group,self.pos,self.dim))

    def named_k(self, k):
        return hax.named(k,(self.batch,self.kpos,self.kv,self.dim)).rearrange((self.batch,self.kv,self.kpos,self.dim))

    def context_array(self, value):
        return value.rearrange((self.batch,self.pos,self.kv,self.group,self.dim)).array.reshape(
            self.config.batch_size,self.config.seq_len,self.config.num_heads,self.dim.size)

    def qk(self, variant, q, k):
        scale = 1.0/math.sqrt(self.dim.size)
        if variant == 'named':
            return hax.dot(self.named_q(q)*scale,self.named_k(k),axis=self.dim).rearrange(
                (self.batch,self.kv,self.group,self.pos,self.kpos)).array
        q = q.reshape(q.shape[0],q.shape[1],self.kv.size,self.group.size,self.dim.size)
        return jnp.einsum('bqhgd,bkhd->bhgqk',q*scale,k)

    def softmax(self, variant, scores):
        if variant == 'named':
            weights = hax.named(scores,(self.batch,self.kv,self.group,self.pos,self.kpos))
            mask = NamedMask.causal().materialize(self.pos,self.kpos)
            return hax.nn.softmax(hax.where(mask,weights,-1e9),axis=self.kpos).array
        mask = jnp.arange(self.config.seq_len)[:,None] >= jnp.arange(self.config.seq_len)[None,:]
        masked = jnp.where(mask,scores,jnp.asarray(-1e9,dtype=scores.dtype))
        return jax.nn.softmax(masked,axis=-1).astype(scores.dtype)

    def pv(self, variant, weights, v):
        if variant == 'named':
            weights = hax.named(weights,(self.batch,self.kv,self.group,self.pos,self.kpos))
            return self.context_array(hax.dot(weights,self.named_k(v),axis=self.kpos))
        context = jnp.einsum('bhgqk,bkhd->bqhgd',weights,v)
        return context.reshape(context.shape[0],context.shape[1],self.config.num_heads,self.dim.size).astype(v.dtype)

    def full(self, variant, q, k, v):
        if variant == 'named':
            return self.context_array(simple_attention_with_dropout(self.pos,self.kpos,self.dim,
                self.named_q(q),self.named_k(k),self.named_k(v),NamedMask.causal(),attention_dtype=q.dtype))
        return reference_attention(q,k,v,AttentionMask.causal(),logits_dtype=None)

    def rope(self, variant, value):
        positions = jnp.asarray(self.positions)
        if variant == 'named':
            heads = hax.Axis('heads',value.shape[2])
            named = hax.named(value,(self.batch,self.pos,heads,self.dim))
            rotary = DefaultRotaryEmbeddingsConfig(theta=10000.0,factor=1.0).build(self.dim)
            return rotary(named,hax.named(positions,(self.batch,self.pos))).rearrange((self.batch,self.pos,heads,self.dim)).array
        return apply_rotary(value,positions,RotaryConfig(theta=10000.0,factor=1.0))

    def function(self, variant, stage):
        if stage in ('rope_q','rope_k'):
            return lambda x:self.rope(variant,x)
        if stage == 'composed':
            return lambda q,k,v:self.pv(variant,self.softmax(variant,self.qk(variant,q,k)),v)
        fn = getattr(self,stage)
        return lambda *xs:fn(variant,*xs)

    def run(self, label, variant, stage, arguments, specs, cotangent, output_spec):
        mesh = self.meshes[variant]
        with jax.set_mesh(mesh), hax.axis_mapping(bench._compute_mapping()), jax.default_matmul_precision('highest' if variant=='fp32' else 'default'):
            originals = [host(x) for x in arguments]
            xs = tuple(jax.device_put(x.astype(np.float32) if variant=='fp32' else x,NamedSharding(mesh,spec))
                       for x,spec in zip(originals,specs,strict=True))
            original_cotangent = host(cotangent)
            c = jax.device_put(original_cotangent.astype(np.float32) if variant=='fp32' else original_cotangent,NamedSharding(mesh,output_spec))
            if variant!='fp32' and not np.array_equal(original_cotangent,host(c)):
                raise ValueError('Common cotangent import changed bytes')
            for before,after in zip(originals,xs,strict=True):
                if variant!='fp32' and not np.array_equal(before,host(after)):
                    raise ValueError('Common operand import changed bytes')
            fn = self.function(variant,stage)
            # Match production reference_attention's internal Auto-axis math for standalone contractions.
            if variant=='native' and stage in ('qk','pv','composed'):
                fn = auto_axes(fn,out_sharding=NamedSharding(mesh,output_spec))
            @jax.jit
            def execute(*packed):
                value,backward = jax.vjp(fn,*packed[:-1])
                return value,*backward(packed[-1])
            lowered = execute.lower(*xs,c)
            stablehlo = lowered.as_text().encode()
            compiled = lowered.compile()
            values = compiled(*xs,c)
            jax.block_until_ready(values)
            hlo = compiled.as_text().encode()
            keep = min(len(hlo),HLO_FILE_LIMIT,HLO_TOTAL_LIMIT-self.hlo_bytes)
            if keep <= 0:
                raise ValueError('Diagnostic HLO total size limit exceeded')
            self.hlo_bytes += keep
            filename = f'{label}-{variant}-{stage}.hlo.txt'
            stable_filename = f'{label}-{variant}-{stage}.stablehlo.txt'
            stable_keep = min(len(stablehlo),HLO_FILE_LIMIT,HLO_TOTAL_LIMIT-self.hlo_bytes)
            if stable_keep <= 0:
                raise ValueError('Diagnostic HLO total size limit exceeded')
            self.hlo_bytes += stable_keep
            (self.output/stable_filename).write_bytes(stablehlo[:stable_keep])
            (self.output/filename).write_bytes(hlo[:keep])
            summary = {'matmul_precision':str(jax.config.jax_default_matmul_precision),'arguments':[statistics(x) | {'sharding':str(x.sharding),'partition_spec':str(x.sharding.spec)} for x in xs],
                       'cotangent':statistics(c) | {'sharding':str(c.sharding),'partition_spec':str(c.sharding.spec)},
                       'outputs':[statistics(x) | {'sharding':str(x.sharding),'partition_spec':str(x.sharding.spec)} for x in values],
                       'hlo':{'file':filename,'full_bytes':len(hlo),'saved_bytes':keep,'truncated':keep<len(hlo),
                              'sha256':hashlib.sha256(hlo).hexdigest(),
                              'dot_lines':[line.strip()[:1000] for line in hlo.decode().splitlines() if ' dot(' in line or ' convolution(' in line][:24],
                              'stablehlo_file':stable_filename,'stablehlo_full_bytes':len(stablehlo),'stablehlo_saved_bytes':stable_keep,
                              'stablehlo_truncated':stable_keep<len(stablehlo),
                              'stablehlo_dot_lines':[line.strip()[:1500] for line in stablehlo.decode().splitlines() if 'stablehlo.dot_general' in line][:24]}}
            # Arrays stay in this process; only aggregate JSON and bounded HLO are persisted.
            return tuple(host(x) for x in values),summary

    def compare(self,label,stage,arguments,specs,cotangent,output_spec):
        values,records = {},{}
        for variant in ('named','native','fp32'):
            values[variant],records[variant] = self.run(label,variant,stage,arguments,specs,cotangent,output_spec)
        for left,right in zip(records['named']['arguments']+[records['named']['cotangent']],records['native']['arguments']+[records['native']['cotangent']],strict=True):
            if any(left[key]!=right[key] for key in ('sha256','dtype','shape','partition_spec')):
                raise ValueError('Named/native common operand, cotangent or layout mismatch')
        records['common_bf16_arguments_and_cotangent_exact_with_matching_partition_specs'] = True
        records['comparisons'] = {pair:[difference(a,b) for a,b in zip(values[left],values[right],strict=True)]
            for pair,left,right in [('named_vs_native','named','native'),('named_vs_fp32','named','fp32'),('native_vs_fp32','native','fp32')]}
        write_json(self.output/f'{label}-{stage}.json',records)
        print('REPLAY_BLOCK '+json.dumps({'layer':label,'stage':stage,'named_vs_native':records['comparisons']['named_vs_native']}),flush=True)
        return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=Path(os.environ.get('IRIS_OUTPUT_DIR','/tmp/native-bf16-attention-replay')))
    parser.add_argument('--expected-backend',choices=('cpu','tpu'),required=True)
    args = parser.parse_args()
    if os.environ.get('XLA_FLAGS') != '--xla_allow_excess_precision=false':
        raise ValueError('Use exactly XLA_FLAGS=--xla_allow_excess_precision=false')
    if jax.process_count()!=1 or len(jax.local_devices())!=4 or jax.device_count()!=4:
        raise ValueError('Expected one process with exactly four local/global devices')
    if jax.default_backend()!=args.expected_backend:
        raise ValueError('Unexpected JAX backend')
    if not args.small and (args.expected_backend!='tpu' or any('v5' not in d.device_kind.lower() or 'lite' in d.device_kind.lower() for d in jax.devices())):
        raise ValueError('Full diagnostic requires four TPU v5 devices')
    if os.environ.get('SOURCE_COMMIT',SOURCE)!=SOURCE:
        raise ValueError('Unexpected source pin')
    jax.config.update('jax_default_matmul_precision','default')
    config = bench.BenchmarkConfig(vocab_size=64,hidden_dim=32,intermediate_dim=64,num_layers=2,num_heads=4,
        num_kv_heads=2,batch_size=4,seq_len=8,data_size=2,model_size=2,compute_dtype='bfloat16') if args.small else bench.BenchmarkConfig(
        vocab_size=8192,hidden_dim=1024,intermediate_dim=2816,num_layers=8,num_heads=16,num_kv_heads=4,
        batch_size=8,seq_len=512,data_size=2,model_size=2,compute_dtype='bfloat16')
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a fresh empty output directory')
    replay = Replay(config,output)
    mesh = replay.meshes['named']
    tokens_host = np.random.default_rng(config.seed+2).integers(0,config.vocab_size,(config.batch_size,config.seq_len),dtype=np.int32)
    weights_host = np.ones_like(tokens_host,dtype=np.float32); weights_host[:,-1]=0
    with jax.set_mesh(mesh),hax.axis_mapping(bench._parameter_mapping()):
        model = hax.named_jit(bench._legacy_config(config).build,axis_resources=bench._parameter_mapping())(
            hax.Axis('vocab',config.vocab_size),key=jax.random.PRNGKey(config.seed))
        parameter_hash = bench._tensor_digest(bench._canonical_parameters(model))
        model = jax.tree.map(lambda x:x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x,model)
        tokens = jax.device_put(tokens_host,NamedSharding(mesh,P('data',None)))
        loss_weights = jax.device_put(weights_host,NamedSharding(mesh,P('data',None)))
        named_tokens = hax.named(tokens,(replay.batch,replay.pos))
        positions = hax.named(jax.device_put(replay.positions,NamedSharding(mesh,P('data',None))),(replay.batch,replay.pos))
        @eqx.filter_jit
        def inputs_and_hidden(m,t):
            x = m.embeddings.embed(t)
            first = x
            for layer in m.transformer.layers.unstacked()[:-1]:
                x = layer(x,NamedMask.causal(),pos_ids=positions)
            last = x
            x = m.transformer.layers.unstacked()[-1](x,NamedMask.causal(),pos_ids=positions)
            return first.array,last.array,m.transformer.norm(x).array
        with hax.axis_mapping(bench._compute_mapping()):
            first,last,captured_hidden = inputs_and_hidden(model,named_tokens)
            normal_hidden = eqx.filter_jit(lambda m,t:m.activations(t,NamedMask.causal(),pos_ids=positions))(model,named_tokens).array
            prefix_check = difference(normal_hidden,captured_hidden)
            named_batch = LmExample(named_tokens,hax.named(loss_weights,(replay.batch,replay.pos)),NamedMask.causal())
            normal_loss = eqx.filter_jit(bench._legacy_loss)(model,named_batch,key=None)
        source_hashes = {}
        for relative in ['lib/levanter/src/levanter/grug/attention/_core.py','lib/levanter/src/levanter/layers/attention.py',
                         'lib/levanter/src/levanter/layers/rotary_native.py','lib/levanter/src/levanter/layers/rotary.py',
                         'lib/levanter/src/levanter/models/llama_native.py','lib/levanter/scripts/bench/bench_llama_native.py']:
            source_hashes[relative] = hashlib.sha256(Path(relative).read_bytes()).hexdigest()
        provenance = {'schema':'native-bf16-common-attention.v1','source_commit':SOURCE,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_file_sha256':source_hashes,'config':dataclasses.asdict(config),'parameter_sha256':parameter_hash,
            'batch_sha256':bench._tensor_digest({'tokens':tokens_host,'loss_weight':weights_host}),
            'devices':[{'id':d.id,'kind':d.device_kind,'process_index':d.process_index,'description':str(d),'coordinates':list(d.coords) if d.platform=='tpu' else None,'core_on_chip':d.core_on_chip if d.platform=='tpu' else None} for d in jax.devices()],
            'process_count':jax.process_count(),'global_device_count':jax.device_count(),'local_device_count':jax.local_device_count(),
            'jax':jax.__version__,'xla_flags':os.environ['XLA_FLAGS'],'matmul_precision':str(jax.config.jax_default_matmul_precision),
            'ordinary_scan_vs_unrolled_capture_final_hidden':prefix_check,'ordinary_scan_loss':float(normal_loss),
            'variant_matmul_precision':{'named':'default','native':'default','fp32':'highest'},
            'limitations':'Unrolled ordinary layer calls and returned intermediates can change fusion; this is common-operand localization, not full Trainer parity/timing. No model/optimizer update or production patch.'}
        write_json(output/'provenance.json',provenance)
        for index,x in [(0,first),(config.num_layers-1,last)]:
            label=f'layer-{index}'
            @eqx.filter_jit
            def operands(m,x):
                layer=m.transformer.layers.unstacked()[index]
                normalized=layer.input_layernorm(hax.named(x,(replay.batch,replay.pos,replay.embed)))
                ac=layer.self_attn.config
                q=layer.self_attn.q_proj(normalized).rearrange((replay.batch,replay.pos,ac.KVHeads,ac.QHeadsPerGroup,ac.HeadSize))
                k=layer.self_attn.k_proj(normalized).rearrange((replay.batch,replay.pos,ac.KVHeads,ac.HeadSize))
                v=layer.self_attn.v_proj(normalized).rearrange((replay.batch,replay.pos,ac.KVHeads,ac.HeadSize)).array
                qr=layer.self_attn.rot_embs(q,positions).rearrange(q.axes).array.reshape(config.batch_size,config.seq_len,config.num_heads,replay.dim.size)
                kr=layer.self_attn.rot_embs(k,positions).rearrange(k.axes).array
                return q.array.reshape(qr.shape),k.array,qr,kr,v
            with hax.axis_mapping(bench._compute_mapping()):
                qpre,kpre,q,k,v=operands(model,x)
                context = replay.full('named',q,k,v)
                @eqx.filter_jit
                def context_cotangent(m,c,x):
                    def suffix(c):
                        layers=m.transformer.layers.unstacked()
                        layer=layers[index]
                        named=hax.named(c.reshape(config.batch_size,config.seq_len,config.num_heads,replay.dim.size),
                            (replay.batch,replay.pos,layer.self_attn.config.Heads,replay.dim))
                        h=hax.named(x,(replay.batch,replay.pos,replay.embed))+layer.self_attn.o_proj(named)
                        h=h+layer.mlp(layer.post_attention_layernorm(h))
                        for following in layers[index+1:]:
                            h=following(h,NamedMask.causal(),pos_ids=positions)
                        h=m.transformer.norm(h).rearrange((replay.batch,replay.pos,replay.embed)).array
                        head=m.get_lm_head().rearrange(('vocab','embed')).array
                        return bench._shared_loss(h,head,tokens,loss_weights)
                    return jax.value_and_grad(suffix)(c)
                loss,ct=context_cotangent(model,context,x)
                # The same common cotangent drives both variants, not independently generated gradients.
                full=replay.compare(label,'full',(q,k,v),(replay.spec,)*3,ct,replay.spec)
                scores=replay.qk('named',q,k)
                weights=replay.softmax('named',scores)
                pv=replay.compare(label,'pv',(weights,v),(replay.score_spec,replay.spec),ct,replay.spec)
                softmax=replay.compare(label,'softmax',(scores,),(replay.score_spec,),pv['named'][1],replay.score_spec)
                qk=replay.compare(label,'qk',(q,k),(replay.spec,)*2,softmax['named'][1],replay.score_spec)
                replay.compare(label,'rope_q',(qpre,),(replay.spec,),full['named'][1],replay.spec)
                replay.compare(label,'rope_k',(kpre,),(replay.spec,),full['named'][2],replay.spec)
                composed=replay.compare(label,'composed',(q,k,v),(replay.spec,)*3,ct,replay.spec)
                write_json(output/f'{label}-capture.json',{'source_suffix_loss':float(loss),'input':statistics(x),
                    'q_pre':statistics(qpre),'k_pre':statistics(kpre),'q':statistics(q),'k':statistics(k),'v':statistics(v),
                    'context_cotangent':statistics(ct),'suffix_vs_ordinary_scan_loss':difference(normal_loss,loss),
                    'split_vs_full_context':difference(full['named'][0],pv['named'][0]),
                    'separately_staged_vs_full_named_vjp':[difference(a,b) for a,b in zip(full['named'][1:],(qk['named'][1],qk['named'][2],pv['named'][2]),strict=True)],
                    'full_vs_recomposed_forward_and_vjp':{variant:[difference(a,b) for a,b in zip(full[variant],composed[variant],strict=True)] for variant in ('named','native','fp32')}})
    write_json(output/'outcome.json',{'completed':True,'diagnostic_only':True,'layers':[0,config.num_layers-1],
        'hlo_bytes':replay.hlo_bytes,'no_tensor_or_checkpoint_artifacts':True})
    print('REPLAY_COMPLETE '+str(output),flush=True)

if __name__=='__main__':
    main()
