"""Build the one authorized A11 native root, without submitting."""
from pathlib import Path
import copy,hashlib,json
from experiments.post_training.identity_path_b.ablation_launch import submission
HERE=Path(__file__).resolve().parent

def request(credentials:dict[str,str])->dict:
    receipt=json.loads((HERE/'review-receipt.json').read_text())
    for path,item in receipt['files'].items():
        raw=Path(path).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('A11 immutable packet mismatch: '+path)
    plan=json.loads((HERE/'launch-plan.json').read_text());target=plan['arms'][0]
    if target['arm']!='A11' or len(plan['arms'])!=1:raise ValueError('Only A11 belongs to this execution packet')
    training=copy.deepcopy(target['training_kwargs']);training['runtime_bytes']=Path(plan['runtime']['local']).read_bytes()
    return submission(name=target['root_name'],training=training,credentials=credentials)
