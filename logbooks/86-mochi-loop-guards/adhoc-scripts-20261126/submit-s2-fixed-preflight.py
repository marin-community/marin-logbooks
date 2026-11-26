import subprocess
program=r'''
import contextlib,io,json,os,yaml
from dataclasses import replace
from marin.execution.lazy import materialized_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from experiments.post_training import rl,rl_compose
from experiments.post_training.mochi_loop_guards.__main__ import GUARDS
from experiments.post_training.mochi_loop_guards.presets import S2_CONTROL,S2_TREATMENT,POOL_SELECTION
configure_coreweave_s3()
rl_compose.SELECTION_ROOT='s3://marin-us-east-08a/marin/rl-data/task-selection'
os.environ['MARIN_PREFIX']='s3://marin-us-east-08a/tmp/ttl=3d/marin'
os.environ['MARIN_TEMP_PREFIX']='s3://marin-us-east-08a/tmp/ttl=3d'
out=StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/s2-fixed-n4-preflight')
plans={};results=[]
for base in (S2_CONTROL,S2_TREATMENT):
 p=replace(base.with_updates(4),revision='2026.11.26.3')
 table=replace(GUARDS,presets=(p,))
 log=io.StringIO()
 with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  code=rl.main(table,argv=['--preset',p.name,'--version',p.revision,'--tasks',POOL_SELECTION,'--plan'])
 (out/(p.name+'-plan.log')).write_text(log.getvalue())
 if code:raise RuntimeError(p.name+' native plan failed; see regional plan log')
 spec=rl_compose.run_spec(p,name=p.name,model_name='mochi',tasks=rl_compose.selection_artifact(POOL_SELECTION))
 step=rl_compose.run_step(p,spec)
 source=materialized_config(step,os.environ['MARIN_PREFIX']).launch_config_yaml
 (out/(p.name+'.yaml')).write_text(source)
 d=yaml.safe_load(source);plans[p.name]=d;g=d['skyrl']['generator'];a=d['iris']['allocation']
 assert a['num_nodes']*a['gpus_per_node']==60
 assert a['memory']=='704GiB' and d['iris']['priority']=='production' and not d['iris'].get('excluded_nvlink_domains')
 assert g['model_dtype']=='bfloat16' and g['inference_engine_tensor_parallel_size']==1 and g['inference_engine_data_parallel_size']==12
 assert g['engine_init_kwargs']['compilation_config']['cudagraph_mode']=='PIECEWISE'
 assert d['skyrl']['trainer']['max_steps']==4 and p.checkpoint_every==p.keep_every==1
 assert p.recipe.layout.workload.plan.train_batch_size==32 and g['n_samples_per_prompt']==8
 results.append({'arm':p.name,'N':4,'N_fixed':True,'revision':p.revision,'native_cli_plan_exit':code,'allocation':a,'antithrash':g['trajectory_reward_shaping']['antithrash'],'CUDA_graphs':'PIECEWISE','TP':1,'model_dtype':'bfloat16','priority':'production','all_racks_eligible':True})
 print(json.dumps(results[-1]),flush=True)
c=plans[S2_CONTROL.name];t=plans[S2_TREATMENT.name]
assert c['inputs']==t['inputs'] and c['iris']['allocation']==t['iris']['allocation']
cs=json.loads(json.dumps(c['skyrl']));ts=json.loads(json.dumps(t['skyrl']))
ca=cs['generator']['trajectory_reward_shaping'].pop('antithrash');ta=ts['generator']['trajectory_reward_shaping'].pop('antithrash')
assert cs==ts
assert ca['enabled'] is False and ta['enabled'] is True and ta['applies_to']=='correct_only'
receipt={'status':'FIXED_BEFORE_SUBMISSION_USER_OVERRIDE','N':4,'N_fixed':True,'GPU_submitted':False,'RL_training_started':False,'model_trials_created':0,'source_commit':'855c6c4c4f','real_regional_pool':POOL_SELECTION,'native_cli_plans_pass':True,'scientific_difference_only':'generator.trajectory_reward_shaping.antithrash','same_inputs_and_all_other_skyrl_fields':True,'checkpoint_every':1,'keep_every':1,'arms':results}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(out/'receipt.json'),'preflight_only':True,'GPU_submitted':False}),flush=True)
'''
compile(program,'regional-conditional-s2-preflight','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','7200','--sync-package','marin-core','--job-name','mochi-loop-guards-s2-fixed-n4-plan-20261126b','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(args).returncode)
