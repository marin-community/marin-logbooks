from pathlib import Path
import copy,hashlib,json
from experiments.post_training.identity_path_b.ablation_launch import submission
HERE=Path(__file__).resolve().parent

def request(credentials:dict[str,str])->dict:
 receipt=json.loads((HERE/'review-receipt.json').read_text())
 for path,item in receipt['files'].items():
  raw=Path(path).read_bytes()
  if len(raw)!=item['bytes']or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('A13R1 sealed input changed: '+path)
 plan=json.loads((HERE/'launch-plan.json').read_text());arm=plan['arm'];training=copy.deepcopy(arm['training_kwargs']);training['runtime_bytes']=Path(plan['runtime']['local']).read_bytes()
 return submission(name=arm['root_name'],training=training,credentials=credentials)
