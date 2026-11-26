import os,subprocess
from pathlib import Path
credential=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret','DAYTONA_RL_API_KEY'],capture_output=True,text=True)
if credential.returncode:raise RuntimeError('RL secret resolution failed')
os.environ['DAYTONA_RL_API_KEY']=credential.stdout.strip()
r=subprocess.run(['uv','run','--project','lib/shellbox','--frozen','--extra','daytona','shellbox-daytona-snapshots','--api-key-env','DAYTONA_RL_API_KEY','purge'],capture_output=True,text=True,timeout=120)
Path('/tmp/mochi-loop-guards/s2-snapshot-cleanup-preview-2308.txt').write_text(r.stdout)
if r.returncode:raise RuntimeError('Read-only snapshot cleanup preview failed')
from collections import Counter
print(dict(Counter(line.split()[0] for line in r.stdout.splitlines() if line.strip())))
