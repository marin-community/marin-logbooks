"""Construct one in-region CPU archive job, never submit or delete."""
from pathlib import Path
import hashlib,json,subprocess
from experiments.post_training.identity_path_b.hf_archive import submission
HERE=Path(__file__).resolve().parent

def request(credentials:dict[str,str])->dict:
    receipt=json.loads((HERE/'receipt.json').read_bytes())
    for item in receipt['files']:
        raw=Path(item['path']).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Private archive sealed file changed: '+item['path'])
    plan=json.loads((HERE/'launch-plan.json').read_bytes());root=Path(plan['workspace'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=plan['code_pin'] or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('Private archive workspace is not the sealed clean source')
    return submission(credentials)
