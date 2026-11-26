import subprocess
program=r'''
import collections,json
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.redaction import redact_string
from vendor.skyrl.scripts.loop_audit import read_rows,trial_turns,verdict_passed
configure_coreweave_s3()
root='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.26-v12'
manifest=json.loads((StoragePath(root)/'resolved-manifest.json').read_text())
membership={r['task_id']:c for c,d in manifest['cohorts'].items() for r in d['rows']}
counts={c:collections.Counter() for c in manifest['cohorts']};samples={c:[] for c in manifest['cohorts']}
for row in read_rows('s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-8d1a0b19d0-in-af5ab9b8/2026.11.25.9/exports/**/dumped_evals/*/eval_only/*.jsonl'):
 c=membership.get(row.get('task_id'))
 if c is None:continue
 turns=trial_turns(row.get('messages') or [])
 gaps=[(i,t) for i,t in enumerate(turns[:-1]) if not t.observation]
 for i,t in gaps:
  counts[c]['gaps']+=1
  counts[c]['tool_or_terminus_action_gaps']+=t.action.startswith(('tool','terminus'))
 if gaps and len(samples[c])<3:
  samples[c].append({'task_id':row['task_id'],'uid':row.get('uid'),'turns':len(turns),'graded':verdict_passed(row) is not None,'gaps':[{'index':i,'action':redact_string(t.action[:1200]),'message':redact_string(t.message[:3000]),'next_action':redact_string(turns[i+1].action[:1200])} for i,t in gaps[:5]]})
report={'model_trials_created':0,'purpose':'Inspect why aggregate observation count is below assistant-turn gap count; helper turns may not have terminal feedback','counts':counts,'samples':samples}
output=root+'/feedback-gap-review.json';StoragePath(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':output,'counts':counts}),flush=True)
'''
compile(program,'gap-review','exec')
outer='import subprocess\nfrom marin.rl.skyrl import skyrl_cpu_command\nsubprocess.run(skyrl_cpu_command("python","-c",'+repr(program)+'),check=True)'
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-v12-feedback-gap-review-20261126a','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(args).returncode)
