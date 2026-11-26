import subprocess
import textwrap

program=textwrap.dedent('''
import json
from collections import Counter
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from vendor.skyrl.scripts.loop_audit import read_rows,is_infrastructure_error
configure_coreweave_s3()
source='s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-f28d12bf4c-in-af5ab9b8/2026.11.25.9/exports/dumped_evals/a54355029268/eval_only/DCAgent2_terminal_bench_2.jsonl'
rows=[]
errors=Counter();statuses=Counter();seen=Counter();keys=set()
for saved in read_rows(source):
    keys.update(saved)
    task=saved['task_id'];repetition=seen[task];seen[task]+=1
    status=(saved.get('verification') or {}).get('status');statuses[str(status)]+=1
    if not is_infrastructure_error(saved):continue
    errors[str(saved.get('exception_type'))]+=1
    rows.append({key:saved.get(key) for key in ('task_id','uid','ordinal','attempt_id','exception_type','error_treatment','stop_reason')} | {'repetition_in_saved_order':repetition,'verification_status':status,'response_tokens':len(saved.get('response_token_ids') or []),'server_error_excerpt':redact_string(str(saved.get('server_error')))[:400], 'verification_excerpt':redact_string(json.dumps(saved.get('verification')))[:3000], 'timing': {key:value for key,value in (saved.get('env_metrics') or {}).items() if key.startswith('timing/')}})
result={'source':source,'saved_trials':sum(seen.values()),'row_fields':sorted(keys),'infrastructure_error_counts':dict(errors),'verification_status_counts':dict(statuses),'infrastructure_trials':rows}
StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/partial/tb2/error-census-details.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps({'saved_trials':sum(seen.values()),'infrastructure_error_counts':dict(errors),'verification_status_counts':dict(statuses)}))
''')
compile(program,'tb2-error-census','exec')
outer='from marin.rl.skyrl import skyrl_cpu_command; import subprocess; subprocess.run(skyrl_cpu_command("python","-c",'+repr(program)+'),check=True)'
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','4','--memory','16GB','--disk','64GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-tb2-errors-20261125b','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(args).returncode)
