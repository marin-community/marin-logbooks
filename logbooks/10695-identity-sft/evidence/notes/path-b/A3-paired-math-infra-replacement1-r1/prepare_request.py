"""Build the one A3 math-only owner; never submits or calls a model."""
from pathlib import Path
import hashlib,json
from experiments.post_training.identity_fullscreen_owner.math_only import submission
HERE=Path(__file__).resolve().parent

def request(arm:str,credentials:dict[str,str])->dict:
    for item in json.loads((HERE/'receipt.json').read_text())['files']:
        raw=Path(item['path']).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('A3 paired-math sealed file changed: '+item['path'])
    plan=json.loads((HERE/'launch-plan.json').read_text());target=plan['targets'][arm]
    archive=Path(plan['source_archive']['path']).read_bytes()
    return submission(name=target['root_name'],candidate=target['candidate'],archive=archive,credentials=credentials)
