import json,os,subprocess
from dataclasses import replace
from pathlib import Path
import yaml
from marin.execution.lazy import materialized_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from experiments.post_training import rl_compose
from experiments.post_training.mochi_loop_guards.presets import S1_STOP,S1_RP
configure_coreweave_s3()
root='s3://marin-us-east-08a/marin'
rl_compose.SELECTION_ROOT=root+'/rl-data/task-selection'
os.environ['MARIN_PREFIX']=root
os.environ['MARIN_TEMP_PREFIX']='s3://marin-us-east-08a/tmp/ttl=3d'
results=[]
for base in (S1_STOP,S1_RP):
 preset=replace(base,revision="2026.11.26.2")
 spec=rl_compose.run_spec(preset,name=preset.name,model_name='mochi')
 step=rl_compose.run_step(preset,spec)
 source=materialized_config(step,root).launch_config_yaml
 Path('/tmp/mochi-loop-guards/'+preset.name+'-repeat1-native-launch.yaml').write_text(source)
 doc=yaml.safe_load(source)
 allocation=doc['iris']['allocation'];gen=doc['skyrl']['generator']
 assert allocation['num_nodes']*allocation['gpus_per_node']==64
 assert allocation['memory']=='128GiB' and doc['iris']['priority']=='production'
 assert not doc['iris'].get('excluded_nvlink_domains')
 assert gen['model_dtype']=='bfloat16' and gen['inference_engine_tensor_parallel_size']==1
 assert gen['engine_init_kwargs']['compilation_config']['cudagraph_mode']=='PIECEWISE'
 item={'preset':preset.name,'revision':preset.revision,'gpu_submitted':False,'allocation':allocation,'priority':doc['iris']['priority'],'all_racks_eligible':True,'model_dtype':gen['model_dtype'],'tensor_parallel':gen['inference_engine_tensor_parallel_size'],'inference_engines':gen['num_inference_engines'],'eval_sampling':gen['eval_sampling_params'],'cuda_graphs':gen['engine_init_kwargs']['compilation_config']['cudagraph_mode'],'followup_trials':723,'swe100_excluded_by_registered_budget_drop':True}
 results.append(item)
receipt={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),'native_cli_plans_exit_code':{'s1-stop':0,'s1-rp':0},'arms':results}
Path('/tmp/mochi-loop-guards/s1-repeat1-preflight-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
