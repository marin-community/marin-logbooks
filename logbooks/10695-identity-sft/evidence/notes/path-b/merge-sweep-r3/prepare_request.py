"""Read-only preparation of one sealed CPU merge submission."""
from pathlib import Path
import hashlib,json,subprocess
from experiments.weight_merging.path_b_sweep.worker import submission
HERE=Path(__file__).resolve().parent

def request(label:str,credentials:dict[str,str])->dict:
 receipt=json.loads((HERE/'review-receipt.json').read_text())
 for path,item in receipt['files'].items():
  raw=Path(path).read_bytes()
  if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Sealed merge binding differs: '+path)
 plan=json.loads((HERE/'launch-plan.json').read_text())
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan['workspace'],text=True).strip()!=plan['code_pin']:raise ValueError('Merge source pin differs')
 if subprocess.check_output(['git','status','--porcelain'],cwd=plan['workspace'],text=True).strip():raise ValueError('Merge checkout is not clean')
 return submission(label,plan['code_pin'],credentials)
