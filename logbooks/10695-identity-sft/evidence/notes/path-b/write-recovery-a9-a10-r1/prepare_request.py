"""Construct the two explicitly authorized recoveries; never submit or probe."""
from pathlib import Path
import hashlib,json
from experiments.post_training.identity_path_b.write_recovery import recovery_submission
HERE=Path(__file__).resolve().parent

def request(label,credentials):
    receipt=json.loads((HERE/'review-receipt.json').read_bytes())
    for name,item in receipt['files'].items():
        raw=Path(name).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Recovery packet changed: '+name)
    plan=json.loads((HERE/'launch-plan.json').read_bytes())
    if label not in plan['arms']:raise ValueError('Only A9 export-only and A10 fresh retrain are authorized')
    runtime=Path(plan['runtime']['local']).read_bytes()
    if hashlib.sha256(runtime).hexdigest()!=plan['runtime']['sha256']:raise ValueError('Original runtime changed')
    return recovery_submission(label=label,runtime_bytes=runtime,credentials=credentials)
