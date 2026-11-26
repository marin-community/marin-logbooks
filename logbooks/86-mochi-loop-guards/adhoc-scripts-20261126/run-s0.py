import os
from pathlib import Path
import subprocess
import sys

for target, secret in [('WANDB_API_KEY','BELAY_WANDB_API_KEY'),('DAYTONA_EVAL_API_KEY','DAYTONA_EVAL_API_KEY')]:
    result=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret',secret],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f'{target} resolution failed')
    os.environ[target]=result.stdout.strip()
os.environ['HF_TOKEN']=Path.home().joinpath('.cache/huggingface/token').read_text().strip()
os.environ['WANDB_ENTITY']='marin-community'
os.execvp('uv',['uv','run','--no-sync','python','-m','experiments.post_training.mochi_loop_guards','--preset','s0-baseline','--version','2026.11.25.9','--issue','12264','--run','--wait'])
