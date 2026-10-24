import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import jmp
import numpy as np
import optax
import haliax as hax

sys.path.insert(0, str(Path('lib/levanter/scripts/bench').resolve()))
import bench_llama_native as bench
from levanter.data.text.examples import GrugLmExample
from levanter.layers.attention import AttentionMask
from levanter.models.llama_native import NativeLlamaModel, NativeLlamaSharding
from levanter.models.lm_model import LmExample
from levanter.tracker import NoopTracker, current_tracker
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig

SMALL = '--small' in sys.argv
CONFIG = bench.BenchmarkConfig(vocab_size=64, hidden_dim=32, intermediate_dim=64, num_layers=2,
    num_heads=4, num_kv_heads=2, batch_size=4, seq_len=16, data_size=2, model_size=2) if SMALL else bench.BenchmarkConfig(
    vocab_size=65536, hidden_dim=2048, intermediate_dim=5632, num_layers=28, num_heads=16,
    num_kv_heads=4, batch_size=8, seq_len=2048, data_size=2, model_size=2)
TARGET = 2 if SMALL else 25
OUTPUT = Path(os.environ.get('IRIS_OUTPUT_DIR', '/private/tmp/native-jax-common-state-cpu'))
OUTPUT.mkdir(parents=True, exist_ok=True)


def host(tree):
    return jax.tree.map(np.array, tree)


def canonical(model):
    return host(bench._canonical_parameters(model))


def trainer(kind, mesh):
    cfg = TrainerConfig(id=f'common-state-{kind}', mp=jmp.Policy(param_dtype=jnp.float32,
        compute_dtype=jnp.float32, output_dtype=jnp.float32), train_batch_size=CONFIG.batch_size,
        num_train_steps=TARGET+2, mesh=MeshConfig(axes={'data':2,'model':2},
        compute_mapping=bench._compute_mapping(), param_mapping=bench._parameter_mapping()),
        use_explicit_mesh_axes=kind=='native', load_checkpoint=False, log_jaxprs=False,
        log_xla_hlo=False, tracker=())
    optim=optax.adamw(CONFIG.learning_rate,b1=CONFIG.beta1,b2=CONFIG.beta2,eps=CONFIG.epsilon,weight_decay=CONFIG.weight_decay)
    result=Trainer(cfg,optim,bench._legacy_loss if kind=='legacy' else bench._native_loss,
        add_default_hooks=False, explicit_layout=ExplicitTrainingLayout(mesh,P('data',None)) if kind=='native' else None)
    result.add_hook(bench._CaptureStep(),every=1)
    return result


def native_state(template, source, mesh, source_models):
    # Host snapshots cross Auto/Explicit mesh modes without retaining donated aliases.
    model=template.model.from_state_dict(source_models[0])
    mu=template.opt_state[0].mu.from_state_dict(source_models[1])
    nu=template.opt_state[0].nu.from_state_dict(source_models[2])
    scalar=lambda value:jax.device_put(np.array(value),NamedSharding(mesh,P()))
    adam=template.opt_state[0]._replace(mu=mu,nu=nu,count=scalar(source.opt_state[0].count))
    result=dataclasses.replace(template,model=model,opt_state=(adam,*template.opt_state[1:]),
        step=scalar(source.step),training_key=scalar(source.training_key))
    jax.block_until_ready(result)
    for label,a,b in [('parameters',source_models[0],result.model),('adam_mu',source_models[1],mu),('adam_nu',source_models[2],nu)]:
        assert bench._tensor_digest(a)==bench._tensor_digest(canonical(b)),label
    assert int(source.opt_state[0].count)==int(result.opt_state[0].count)
    assert np.array_equal(source.training_key,result.training_key)
    return result


def capture(trainer, state, batch):
    diag=trainer._jit_train_step_fn(bench._copy_arrays(state),(batch,),{})
    jax.block_until_ready(diag)
    grads,updates=diag.hook_infos[0]
    values={'loss':{'loss':np.array(diag.loss)},'gradients':canonical(grads),'updates':canonical(updates),
        'parameters_after_one_step':canonical(diag.new_state.model),
        'adam_mu':canonical(diag.new_state.opt_state[0].mu),'adam_nu':canonical(diag.new_state.opt_state[0].nu)}
    del diag,grads,updates
    normal=trainer._jit_train_step_fn_no_hook(bench._copy_arrays(state),(batch,),{})
    jax.block_until_ready(normal)
    values.update(execution_loss={'loss':np.array(normal.loss)},
        execution_parameters_after_one_step=canonical(normal.new_state.model),
        execution_adam_mu=canonical(normal.new_state.opt_state[0].mu),
        execution_adam_nu=canonical(normal.new_state.opt_state[0].nu))
    del normal
    return values


def compare_at(source, step, legacy_trainer, native_trainer, native_template, named_batch, raw_batch, legacy_mesh, native_mesh):
    print(f'COMMON_STATE checkpoint={step} capture_start',flush=True)
    with jax.set_mesh(legacy_mesh),hax.axis_mapping(bench._compute_mapping()):
        initial=canonical(source.model)
        initial_mu=canonical(source.opt_state[0].mu)
        initial_nu=canonical(source.opt_state[0].nu)
        legacy_values=capture(legacy_trainer,source,named_batch)
    with jax.set_mesh(native_mesh):
        matched=native_state(native_template,source,native_mesh,(initial,initial_mu,initial_nu))
        native_values=capture(native_trainer,matched,raw_batch)
        del matched
    parity={label:bench._numerical_comparison(legacy_values[label],native_values[label],CONFIG) for label in legacy_values}
    acceptance=bench.numerical_acceptance(parity,CONFIG)
    # Include bounded coordinates that distinguish gradient amplification from optimizer arithmetic.
    names=sorted(parity['updates']['tensors'],key=lambda name:parity['updates']['tensors'][name]['relative_l2'],reverse=True)[:5]
    coordinates=[]
    for name in names:
        a,b=[values['updates'][name] for values in (legacy_values,native_values)]
        indices=np.argsort(np.abs(a-b).reshape(-1))[-8:][::-1]
        for flat in indices:
            ix=np.unravel_index(int(flat),a.shape)
            count=int(source.opt_state[0].count)+1
            record={'tensor':name,'index':[int(v) for v in ix], 'parameter':float(initial[name][ix]),
                'mu_before':float(initial_mu[name][ix]),'nu_before':float(initial_nu[name][ix]),'count_after':count}
            for kind,values in [('legacy',legacy_values),('native',native_values)]:
                gradient=float(values['gradients'][name][ix]);mu=CONFIG.beta1*record['mu_before']+(1-CONFIG.beta1)*gradient
                nu=CONFIG.beta2*record['nu_before']+(1-CONFIG.beta2)*gradient**2
                mu_hat=mu/(1-CONFIG.beta1**count);nu_hat=nu/(1-CONFIG.beta2**count)
                reference=-CONFIG.learning_rate*(mu_hat/(np.sqrt(nu_hat)+CONFIG.epsilon)+CONFIG.weight_decay*record['parameter'])
                record[kind]={'gradient':gradient,'update':float(values['updates'][name][ix]),
                    'mu_after':float(values['adam_mu'][name][ix]),'nu_after':float(values['adam_nu'][name][ix]),
                    'float64_adam_reference':float(reference)}
            coordinates.append(record)
    report={'step_before':step,'common_parameter_sha256':bench._tensor_digest(initial),
        'same_parameters_and_adam_state':True,'parity':parity,'acceptance':acceptance,'worst_update_coordinates':coordinates}
    (OUTPUT/f'common-state-{step}.json').write_text(json.dumps(report,indent=2))
    print('COMMON_STATE_RESULT',json.dumps({'step':step,'loss_legacy':float(legacy_values['execution_loss']['loss']),
        'loss_native':float(native_values['execution_loss']['loss']),'acceptance':acceptance}),flush=True)


def main():
    ds=jax.local_devices()
    assert len(ds)==4
    if not SMALL:assert jax.default_backend()=='tpu' and all('v5' in d.device_kind.lower() and 'lite' not in d.device_kind.lower() for d in ds)
    devices=np.array(ds).reshape(2,2)
    legacy_mesh=Mesh(devices,('data','model'),axis_types=(AxisType.Auto,AxisType.Auto))
    native_mesh=Mesh(devices,('data','model'),axis_types=(AxisType.Explicit,AxisType.Explicit))
    tokens=np.random.default_rng(CONFIG.seed+2).integers(0,CONFIG.vocab_size,(CONFIG.batch_size,CONFIG.seq_len),dtype=np.int32)
    weights=np.ones_like(tokens,dtype=np.float32);weights[:,-1]=0
    with current_tracker(NoopTracker()),jax.set_mesh(legacy_mesh),hax.axis_mapping(bench._parameter_mapping()):
        model=hax.named_jit(bench._legacy_config(CONFIG).build,axis_resources=bench._parameter_mapping())(hax.Axis('vocab',CONFIG.vocab_size),key=jax.random.PRNGKey(CONFIG.seed))
        legacy_trainer=trainer('legacy',legacy_mesh)
        state=legacy_trainer.initial_state(jax.random.PRNGKey(CONFIG.seed+1),model=model)
        spec=NamedSharding(legacy_mesh,P('data',None))
        axes=(hax.Axis('batch',CONFIG.batch_size),hax.Axis('position',CONFIG.seq_len))
        named_batch=LmExample(hax.named(jax.device_put(tokens,spec),axes),hax.named(jax.device_put(weights,spec),axes),AttentionMask.causal())
        initial=canonical(state.model)
    initial_hash=bench._tensor_digest(initial)
    if not SMALL:assert initial_hash=='d79fb8c383f9af209fd9e3e32e00798c4aeb460d779a1444d9ea20ba9f453f9a',initial_hash
    with current_tracker(NoopTracker()),jax.set_mesh(native_mesh):
        model=NativeLlamaModel.init(bench._native_config(CONFIG),key=jax.random.PRNGKey(CONFIG.seed),sharding=NativeLlamaSharding(batch_axes=('data',),fsdp_axis='data',model_axis='model')).from_state_dict(initial)
        native_trainer=trainer('native',native_mesh)
        native_template=native_trainer.initial_state(jax.random.PRNGKey(CONFIG.seed+1),model=model)
        spec=NamedSharding(native_mesh,P('data',None))
        raw_batch=GrugLmExample(jax.device_put(tokens,spec),jax.device_put(weights,spec))
    del model,initial
    provenance={'source_commit':'ad2bc20438796c63415240ed1979e93d212fc6da','diagnostic_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config':dataclasses.asdict(CONFIG),'target_step':TARGET,'initial_parameter_sha256':initial_hash,
        'batch_sha256':bench._tensor_digest({'tokens':tokens,'loss_weight':weights}),
        'devices':[str(d) for d in ds],'jax':jax.__version__,'xla_flags':os.environ.get('XLA_FLAGS'),
        'purpose':'Common-state gradient/update diagnostic; no timing or speed ratio'}
    (OUTPUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    with current_tracker(NoopTracker()):
        compare_at(state,0,legacy_trainer,native_trainer,native_template,named_batch,raw_batch,legacy_mesh,native_mesh)
        losses=[]
        with jax.set_mesh(legacy_mesh),hax.axis_mapping(bench._compute_mapping()):
            for i in range(TARGET):
                result=legacy_trainer._jit_train_step_fn_no_hook(state,(named_batch,),{})
                jax.block_until_ready(result)
                losses.append(float(result.loss));state=result.new_state
                print(f'LEGACY_STEP {i} loss={losses[-1]}',flush=True)
        (OUTPUT/'legacy-losses.json').write_text(json.dumps(losses))
        compare_at(state,TARGET,legacy_trainer,native_trainer,native_template,named_batch,raw_batch,legacy_mesh,native_mesh)
    (OUTPUT/'outcome.json').write_text(json.dumps({'completed':True,'diagnostic_only':True}))

if __name__=='__main__':main()
