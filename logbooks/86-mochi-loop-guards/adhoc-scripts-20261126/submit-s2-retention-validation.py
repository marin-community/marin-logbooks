import subprocess
import textwrap
program=textwrap.dedent(r'''
import json
import os
import subprocess
import sys
from pathlib import Path
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
configure_coreweave_s3()
os.environ['MARIN_PREFIX']='s3://marin-us-east-08a/tmp/ttl=3d/marin'
root=StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-retention')
results=[]
for name in ('s2-antithrash-control','s2-antithrash-treatment'):
    command=[sys.executable,'-m','experiments.post_training.mochi_loop_guards','--preset',name,'--tasks','users/benfeuer/mochi-loop-guards/snowball-100k-agentic-ho-excluded@2026.11.25.1','--version','2026.11.25.7','--plan']
    done=subprocess.run(command,capture_output=True,text=True,timeout=600)
    (root/(name+'-plan.log')).write_text(done.stdout+done.stderr)
    results.append({'arm':name,'exit_code':done.returncode,'updates_placeholder':8,'gpu_submitted':False})
    print(json.dumps(results[-1]),flush=True)
    if done.returncode:
        raise RuntimeError(name+' plan failed; see regional diagnostic log')
(root/'receipt.json').write_text(json.dumps({'source':'acad2fe9ffad994c9bfe3b4f7ffadf5808518d07','results':results,'n_fixed':False,'checkpoint_every':1,'keep_every':1},indent=2)+'\n')
''')
compile(program,'regional-s2-retention-validation','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','32GB','--disk','128GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-s2-retention-check-20261125b','--','uv','run','python','-c',program]
raise SystemExit(subprocess.run(args).returncode)
