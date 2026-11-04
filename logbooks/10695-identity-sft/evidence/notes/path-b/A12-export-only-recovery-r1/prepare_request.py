"""Construct only the authorized production A12 export-only owner; never submit."""
from pathlib import Path
import hashlib,json
from experiments.post_training.identity_path_b.a12_export_recovery import recovery_submission
HERE=Path(__file__).resolve().parent

def request(credentials):
    receipt=json.loads((HERE/'review-receipt.json').read_bytes())
    for name,item in receipt['files'].items():
        raw=Path(name).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('A12 recovery packet changed: '+name)
    plan=json.loads((HERE/'launch-plan.json').read_bytes())
    runtime=Path(plan['runtime']['local']).read_bytes()
    return recovery_submission(runtime_bytes=runtime,credentials=credentials)
