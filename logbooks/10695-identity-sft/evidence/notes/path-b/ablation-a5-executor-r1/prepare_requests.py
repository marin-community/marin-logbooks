"""Construct four authorized capture requests; this module never submits."""
from pathlib import Path
import hashlib,json
from experiments.post_training.identity_path_b.a5.execute import same_task_submission
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'ablation-a5-generation-r1'

def request(shard:int,credentials:dict[str,str])->dict:
    for receipt_path in [PARENT/'receipt.json',HERE/'receipt.json']:
        receipt=json.loads(receipt_path.read_text())
        for item in receipt['files']:
            raw=Path(item['path']).read_bytes()
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise ValueError('A5 bound input differs: '+item['path'])
    plan=json.loads((PARENT/'launch-plan.json').read_text())
    row=next(item for item in plan['shards'] if item['shard']==shard)
    return same_task_submission(row['run_id'],row['artifact_root'],row['manifest']['uri'],row['manifest']['sha256'],shard,credentials)
