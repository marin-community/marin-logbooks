"""Build the authorized A1 export-only CPU request; this module never submits."""
from pathlib import Path
import hashlib,json
from experiments.post_training.identity_path_b.ablation_recovery import recovery_submission
from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent

HERE=Path(__file__).resolve().parent

def request(credentials:dict[str,str])->dict:
    receipt=json.loads((HERE/'receipt.json').read_text())
    for item in receipt['files']:
        raw=Path(item['path']).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Recovery packet hash/length mismatch: '+item['path'])
    plan=json.loads((HERE/'launch-plan.json').read_text())
    runtime=Path(plan['runtime']['local']).read_bytes()
    if hashlib.sha256(runtime).hexdigest()!=plan['runtime']['sha256']:
        raise ValueError('Original runtime bytes differ')
    training={**plan['training_kwargs'],'runtime_bytes':runtime}
    return recovery_submission(name=plan['root_name'],training=training,event=CheckpointEvent(**plan['event']),recovery_root=plan['recovery_root'],evidence=plan['evidence'],credentials=credentials)
