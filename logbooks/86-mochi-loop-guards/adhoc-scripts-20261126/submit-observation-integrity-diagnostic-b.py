from pathlib import Path
import subprocess
from marin.rl.skyrl import skyrl_cpu_command
program=r'''
import collections,json,re
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from vendor.skyrl.scripts.loop_audit import read_rows,trial_turns
configure_coreweave_s3()
root='s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-f28d12bf4c-in-af5ab9b8/2026.11.25.9'
report=[]
for row in read_rows(root+'/exports/**/dumped_evals/*/eval_only/*.jsonl'):
 turns=trial_turns(row.get('messages') or [])
 prompt=row.get('input_prompt','')
 item={'task_id':row['task_id'],'uid':row['uid'],'ordinal':row['ordinal'],'turns':len(turns),'nonempty_observations':sum(bool(t.observation) for t in turns),'prompt_chars':len(prompt),'prompt_special_tokens':dict(collections.Counter(re.findall(r'<\|[^>]+\|>',prompt))),'message_roles':dict(collections.Counter(m.get('role') for m in row.get('messages') or [])),'assistant_fields':sorted({k for m in row.get('messages') or [] if m.get('role')=='assistant' for k in m})}
 if row['task_id'] in ('tblite/legal-summary-extraction','tb2/code-from-image','tb2/qemu-startup'):
  item['prompt_start_excerpt']=redact_string(prompt[:1800]);item['prompt_end_excerpt']=redact_string(prompt[-4000:])
 report.append(item)
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.26/observation-integrity-original.json'
StoragePath(output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'rows':len(report),'output':output,'multi_turn_with_no_observation':sum(x['turns']>1 and not x['nonempty_observations'] for x in report)}))
'''
compile(program,'observation-integrity','exec')
outer='from marin.rl.skyrl import skyrl_cpu_command\nimport subprocess\nsubprocess.run(skyrl_cpu_command("python","-c",'+repr(program)+'),check=True)'
cmd=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-observation-integrity-20261126b','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(cmd).returncode)
