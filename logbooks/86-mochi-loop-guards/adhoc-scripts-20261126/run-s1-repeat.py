"""Run the one registered boundary repeat only after its original arm triggers it."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import subprocess

from experiments.post_training import rl,rl_compose
from experiments.post_training.mochi_loop_guards.__main__ import GUARDS,REGIONAL_ROOT

p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=('stop','rp'));mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--plan',action='store_true');mode.add_argument('--run',action='store_true');a=p.parse_args()
name='s1-'+a.arm
revision='2026.11.26.2'
preset=replace(GUARDS.by_name[name],revision=revision)
table=replace(GUARDS,presets=tuple(preset if row.name==name else row for row in GUARDS.presets))
rl_compose.SELECTION_ROOT=REGIONAL_ROOT+'/rl-data/task-selection'
os.environ['MARIN_PREFIX']=REGIONAL_ROOT
os.environ['MARIN_TEMP_PREFIX']='s3://marin-us-east-08a/tmp/ttl=3d'
os.environ['WANDB_ENTITY']='marin-community'
arguments=['--preset',name,'--version',revision]
if a.run:
 for target,secret in [('WANDB_API_KEY','BELAY_WANDB_API_KEY'),('DAYTONA_EVAL_API_KEY','DAYTONA_EVAL_API_KEY')]:
  r=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret',secret],capture_output=True,text=True)
  if r.returncode:raise RuntimeError(target+' resolution failed')
  os.environ[target]=r.stdout.strip()
 os.environ['HF_TOKEN']=Path.home().joinpath('.cache/huggingface/token').read_text().strip()
 arguments+=['--issue','12264','--run','--wait']
else:arguments+=['--plan']
raise SystemExit(rl.main(table,argv=arguments,forward_env=('MARIN_PREFIX','MARIN_TEMP_PREFIX')))
