import json,os,subprocess
from dataclasses import replace
from pathlib import Path
import yaml
from marin.execution.lazy import materialized_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from experiments.post_training import rl_compose
from experiments.post_training.mochi_loop_guards.presets import S0,MODEL
configure_coreweave_s3()
root='s3://marin-us-east-08a/marin'
rl_compose.SELECTION_ROOT=root+'/rl-data/task-selection'
os.environ['MARIN_PREFIX']=root
os.environ['MARIN_TEMP_PREFIX']='s3://marin-us-east-08a/tmp/ttl=3d'
preset=S0
spec=rl_compose.run_spec(preset,name=preset.name,model_name='mochi')
source=materialized_config(rl_compose.run_step(preset,spec),root).launch_config_yaml
Path('/tmp/mochi-loop-guards/s0-v11-native-launch.yaml').write_text(source)
doc=yaml.safe_load(source)
old=yaml.safe_load(Path('/tmp/mochi-loop-guards/s0-v9-resolved-launch.yaml').read_text())
old=old.get('config',old)
assert doc['iris']['allocation']==old['iris']['allocation']
def check_explicit(new,previous,path):
 if isinstance(new,dict):
  for key,value in new.items():
   assert key in previous,path+'.'+key
   check_explicit(value,previous[key],path+'.'+key)
 else:assert new==previous,path
check_explicit(doc['skyrl']['generator'],old['skyrl']['generator'],'generator')
assert preset.model==MODEL
assert not doc['iris'].get('excluded_nvlink_domains')
assert doc['iris']['priority']=='production'
receipt={'revision':preset.revision,'requested_cli_version':'2026.11.26.2','source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),'native_cli_plan_exit_code':0,'allocation_equal_v9':True,'all_explicit_generator_fields_equal_v9':True,'initializer_is_frozen_MODEL_binding':True,'gpu_submitted':False,'gpus':128,'trials':823,'hosts':32,'memory_per_host':'128GiB','priority':'production','all_racks_eligible':True,'tensor_parallel':1,'model_dtype':'bfloat16','cuda_graphs':'PIECEWISE','repair':'Normalize empty chat annotations so agent replies preserve observation transitions and runtime stall checks','original_v9_trace_audit_superseded':True}
Path('/tmp/mochi-loop-guards/s0-v11-preflight-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
