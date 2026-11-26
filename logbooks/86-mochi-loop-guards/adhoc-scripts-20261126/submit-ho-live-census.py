from pathlib import Path
import subprocess,textwrap
root=Path('/tmp/mochi-loop-guards/s0-v9-trajectory-root.txt').read_text()
program=textwrap.dedent('''
import gzip,io,json,zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
configure_coreweave_s3()
root=TRAJECTORY_ROOT
def summarize(path):
    counts=Counter()
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        for name in archive.namelist():
            if not name.startswith('records/') or not name.endswith('.json.gz'):continue
            record=json.loads(gzip.decompress(archive.read(name)))
            if record['trajectory']['environment_extras'].get('data_source')!='long-loop-smartie-swesmith':continue
            verification=record.get('verification_result') or {}
            status=verification.get('status');error=record['disposition'].get('exception_type')
            reason=redact_string(str(verification.get('reason')))[:300]
            counts[(str(status),str(error),reason)]+=1
    return counts
paths=sorted((StoragePath(root)/'schema_v7/archives/phase=eval/step=eval_only/*.zip').glob(),key=str)
counts=Counter()
with ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(summarize,paths):counts.update(result)
result={'observed_at':datetime.now(timezone.utc).isoformat(),'partial_as_available_not_representative':True,'model_trials_submitted':0,'available_ho_records':sum(counts.values()),'categories':[{'verification_status':status,'exception_type':error,'reason_excerpt':reason,'count':n} for (status,error,reason),n in counts.items()]}
StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/ho-live-census-2055.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
''').replace('root=TRAJECTORY_ROOT','root='+repr(root))
compile(program,'ho-live-census','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','4','--memory','16GB','--disk','64GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-ho-live-census-20261125a','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(args).returncode)
