import json
import os
from dataclasses import replace
from pathlib import Path

import yaml
from iris.cluster.platforms.k8s.coreweave_topology import balanced_rack_slice_size
from marin.execution.lazy import materialized_config
from experiments.post_training import rl, rl_compose
from experiments.post_training.mochi_loop_guards.presets import S0
from experiments.post_training.rl_presets import Generation, ModelPresets

root='s3://marin-us-east-08a/marin'
rl_compose.SELECTION_ROOT=root+'/rl-data/task-selection'
os.environ.setdefault('MARIN_PREFIX',root)
os.environ.setdefault('MARIN_TEMP_PREFIX','s3://marin-us-east-08a/tmp/ttl=3d')
plan=replace(S0.recipe.layout.workload.plan,num_inference_engines=7)
layout=replace(S0.recipe.layout,name='mochi-guards-generate-112',workload=Generation(plan))
preset=replace(S0,name='s0-baseline-112-layout-proposal',recipe=replace(S0.recipe,layout=layout),budget='112 GB200; unchanged 823-trial panel and frozen decoding')
table=ModelPresets(model='mochi',presets=(preset,),namespace='mochi-loop-guards',control_pins={},needs_completed_default_export=False)
code=rl.main(table,argv=['--preset',preset.name,'--version',preset.revision,'--issue','12264','--plan'],entry_module='experiments.post_training.mochi_loop_guards')
if code:raise SystemExit(code)
spec=rl_compose.run_spec(preset,name='s0-112-layout-proposal',model_name='mochi')
step=rl_compose.run_step(preset,spec)
document=materialized_config(step,root).launch_config_yaml
Path('/tmp/mochi-loop-guards/s0-112-proposal.yaml').write_text(document)
receipt={'submitted':False,'gpu_count':plan.num_nodes*plan.gpus_per_node,'nodes':plan.num_nodes,'engines':7,'gpus_per_node':4,'engine_dp':16,'engine_ep':16,'tensor_parallel':1,'dtype':'bfloat16','memory_per_host':layout.host_memory,'priority':layout.priority,'excluded_nvlink_domains':list(layout.excluded_nvlink_domains),'rack_slices':plan.num_nodes//balanced_rack_slice_size(plan.num_nodes),'nodes_per_rack_slice':balanced_rack_slice_size(plan.num_nodes),'trials':823,'master_seed':20261125,'config':'/tmp/mochi-loop-guards/s0-112-proposal.yaml'}
Path('/tmp/mochi-loop-guards/s0-112-proposal-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
