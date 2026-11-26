import json,os
from pathlib import Path
from marin.execution.lazy import materialized_config
from experiments.post_training import rl,rl_compose
from experiments.post_training.mochi_loop_guards.__main__ import GUARDS
from experiments.post_training.mochi_loop_guards.presets import S0_TB2_RECOVERY
root='s3://marin-us-east-08a/marin'
rl_compose.SELECTION_ROOT=root+'/rl-data/task-selection'
os.environ.setdefault('MARIN_PREFIX',root)
os.environ.setdefault('MARIN_TEMP_PREFIX','s3://marin-us-east-08a/tmp/ttl=3d')
preset=S0_TB2_RECOVERY
code=rl.main(GUARDS,argv=['--preset',preset.name,'--version',preset.revision,'--issue','12264','--plan'],entry_module='experiments.post_training.mochi_loop_guards')
if code:raise SystemExit(code)
spec=rl_compose.run_spec(preset,name=preset.name,model_name='mochi')
step=rl_compose.run_step(preset,spec)
Path('/tmp/mochi-loop-guards/s0-tb2-recovery-plan.yaml').write_text(materialized_config(step,root).launch_config_yaml)
print(json.dumps({'native_plan_passed':True,'gpu_submitted':False,'preset':preset.name,'revision':preset.revision}))
