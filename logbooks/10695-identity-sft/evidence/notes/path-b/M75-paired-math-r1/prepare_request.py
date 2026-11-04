"""Build the one M75 math-only owner; never submits or calls a model."""
from pathlib import Path
import hashlib,json
from experiments.post_training.identity_fullscreen_owner.math_only import submission
HERE=Path(__file__).resolve().parent

def request(arm:str,credentials:dict[str,str])->dict:
    for item in json.loads((HERE/'receipt.json').read_text())['files']:
        raw=Path(item['path']).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('M75 paired-math sealed file changed: '+item['path'])
    plan=json.loads((HERE/'launch-plan.json').read_text());target=plan['targets'][arm]
    ready_path=Path(plan['actual_readiness']['path']);ready_raw=ready_path.read_bytes()
    if hashlib.sha256(ready_raw).hexdigest()!=plan['actual_readiness']['sha256']:raise ValueError('Actual same-model readiness changed')
    ready=json.loads(ready_raw)
    if ready['HF']!=target['candidate']['model_path']:raise ValueError('Readiness bound to another model')
    for item in ready['metadata'].values():
        raw=Path(item['path']).read_bytes()
        if len(raw)!=item['bytes']or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Actual model metadata changed')
    archive=Path(plan['source_archive']['path']).read_bytes()
    return submission(name=target['root_name'],candidate=target['candidate'],archive=archive,credentials=credentials)
