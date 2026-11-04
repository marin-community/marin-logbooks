"""Build the one authorized A12/A13 native root, without submitting."""
from pathlib import Path
import copy,hashlib,json
from experiments.post_training.identity_path_b.ablation_launch import submission
HERE=Path(__file__).resolve().parent

def request(label:str,credentials:dict[str,str])->dict:
    receipt=json.loads((HERE/'review-receipt.json').read_text())
    for path,item in receipt['files'].items():
        raw=Path(path).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('A12/A13 immutable packet mismatch: '+path)
    plan=json.loads((HERE/'launch-plan.json').read_text());target=next(a for a in plan['arms'] if a['arm']==label)
    if label not in ('A12','A13') or len(plan['arms'])!=2:raise ValueError('Only A12/A13 belong to this packet')
    training=copy.deepcopy(target['training_kwargs']);training['runtime_bytes']=Path(plan['runtime']['local']).read_bytes()
    return submission(name=target['root_name'],training=training,credentials=credentials)
