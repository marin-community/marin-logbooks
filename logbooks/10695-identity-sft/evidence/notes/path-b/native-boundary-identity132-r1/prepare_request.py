"""Construct three named diagnostic scopes without opening any live client."""
from pathlib import Path
import hashlib,json
from experiments.post_training.identity_path_b.native_boundary_scope.owner import submission
HERE=Path(__file__).resolve().parent

def request(label,credentials):
    for name,item in json.loads((HERE/'review-receipt.json').read_text())['files'].items():
        raw=Path(name).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Native-boundary immutablepacket mismatch: '+name)
    item=json.loads((HERE/'launch-plan.json').read_text())['targets'][label]
    return submission(label=label,name=item['root_name'],output=item['output'],credentials=credentials)
