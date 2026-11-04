"""Construct only the single scoped A6 native root; no live submission."""
from pathlib import Path
import copy,hashlib,json
from experiments.post_training.identity_path_b.ablation_launch import submission
HERE=Path(__file__).resolve().parent

def request(arm:str,credentials:dict[str,str])->dict:
    if arm!='A6':raise ValueError('Only A6 is authorized by this successor')
    receipt=json.loads((HERE/'review-receipt.json').read_text())
    for path,item in receipt['files'].items():
        raw=Path(path).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('A6 immutable packet mismatch: '+path)
    plan=json.loads((HERE/'launch-plan.json').read_text());target=plan['arms'][0]
    training=copy.deepcopy(target['training_kwargs']);training['runtime_bytes']=Path(plan['runtime']['local']).read_bytes()
    return submission(name=target['root_name'],training=training,credentials=credentials)
