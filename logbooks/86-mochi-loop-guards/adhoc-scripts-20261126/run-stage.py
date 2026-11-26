import os
from pathlib import Path
import subprocess
import sys

preset, version = sys.argv[1:3]
if preset not in ('s1-stop', 's1-rp', 's2-antithrash-control', 's2-antithrash-treatment'):
    raise ValueError('Unsupported experiment stage')
secrets = [('WANDB_API_KEY','BELAY_WANDB_API_KEY'), ('DAYTONA_EVAL_API_KEY','DAYTONA_EVAL_API_KEY')]
if preset.startswith('s2-'):
    secrets.append(('DAYTONA_RL_API_KEY','DAYTONA_RL_API_KEY'))
    os.environ['MARIN_PREFIX']='s3://marin-us-east-08a/tmp/ttl=3d/marin'
for target, secret in secrets:
    result=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret',secret],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f'{target} resolution failed')
    os.environ[target]=result.stdout.strip()
os.environ['HF_TOKEN']=Path.home().joinpath('.cache/huggingface/token').read_text().strip()
os.environ['WANDB_ENTITY']='marin-community'
arguments=['uv','run','--no-sync','python','-m','experiments.post_training.mochi_loop_guards','--preset',preset,'--version',version,'--issue','12264','--run','--wait']
if preset.startswith('s2-'):
    arguments.extend(['--tasks','users/benfeuer/mochi-loop-guards/snowball-100k-agentic-ho-excluded@2026.11.25.1'])
os.execvp('uv',arguments)
