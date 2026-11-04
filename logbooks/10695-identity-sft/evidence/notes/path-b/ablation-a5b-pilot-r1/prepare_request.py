"""Construct the one reviewed native request; never submits or reads credentials."""
import hashlib,json
from pathlib import Path
from experiments.post_training.identity_path_b.a5b.execute import submission

HERE=Path(__file__).resolve().parent


def request(credentials:dict[str,str])->dict:
    receipt=json.loads((HERE/'receipt.json').read_text())
    for item in receipt['files']:
        raw=Path(item['path']).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Pilot bound file changed: '+item['path'])
    plan=json.loads((HERE/'launch-plan.json').read_text())
    bound=plan['binding']
    return submission(bound['run_id'],bound['artifact_root'],bound['manifest']['uri'],bound['manifest']['sha256'],0,credentials)
