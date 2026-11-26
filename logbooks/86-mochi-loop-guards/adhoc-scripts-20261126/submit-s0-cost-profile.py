import hashlib,subprocess
program=r'''
import json,hashlib
import numpy as np
from collections import defaultdict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import ATTEMPT_SECONDS_METRIC
from vendor.skyrl.scripts.loop_audit import read_rows,verdict_passed,trial_turns
configure_coreweave_s3()
root='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards'
manifest=json.loads((StoragePath(root)/'audits/s0-2026.11.26-v12/resolved-manifest.json').read_text())
base=json.loads((StoragePath(root)/'audits/s0-2026.11.26-v12/audit.json').read_text())
membership={r['task_id']:name for name,c in manifest['cohorts'].items() for r in c['rows']}
samples=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));row_counts=defaultdict(int)
for row in read_rows('s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-8d1a0b19d0-in-af5ab9b8/2026.11.25.9/exports/**/dumped_evals/*/eval_only/*.jsonl'):
 name=membership.get(row.get('task_id'))
 if name is None:continue
 row_counts[name]+=1
 sequences=row.get('assistant_turn_token_ids')
 if sequences is None:sequences=[row.get('response_token_ids') or []]
 tokens=sum(len(s) for s in sequences)
 wall=(row.get('env_metrics') or {}).get(ATTEMPT_SECONDS_METRIC)
 for scope in ('all_saved', 'graded') if verdict_passed(row) is not None else ('all_saved',):
  samples[name][scope]['assistant_response_tokens'].append(tokens)
  if isinstance(wall,(int,float)):samples[name][scope]['wall_seconds'].append(float(wall))
  samples[name][scope]['assistant_turns'].append(len(trial_turns(row.get('messages') or [])))
def stats(values):
 return {'n':len(values),'sum':float(sum(values)),'mean':float(np.mean(values)),'p50':float(np.percentile(values,50)),'p90':float(np.percentile(values,90)),'p95':float(np.percentile(values,95)),'max':max(values)}
report={'model_trials_created':0,'purpose':'Full per-trial cost distributions from sealed S0; no new selection or detector rule','token_semantics':'sum of all assistant_turn_token_ids; terminal response_token_ids used only when turn arrays absent','wall_semantics':'timing/attempt for each saved attempt, not native phase duration or retry-inclusive wall','cohorts':{name:{scope:{metric:stats(values) for metric,values in metrics.items()} for scope,metrics in scopes.items()} for name,scopes in samples.items()},'saved_rows':row_counts,'native_phase_seconds':{'tb2':2374.555,'tblite':2162,'swe100':2148,'swesmith-ho':10863},'primary_native_vllm_counter':{'generated_tokens':2658970,'wall_seconds':2376.0807617604733,'allocated_gpus':128,'tokens_per_second_per_gpu':8.74263344045967},'native_terminal_turn_response_tokens':49182,'native_terminal_turn_response_tokens_are_not_full_agent_token_cost':True}
print(json.dumps({'median_validation':{name:{'observed':report['cohorts'][name]['graded']['assistant_response_tokens']['p50'],'sealed_audit':base['agents'][name]['cost']['tokens_p50'],'graded_n':report['cohorts'][name]['graded']['assistant_response_tokens']['n']} for name in manifest['cohorts']}}),flush=True)
for name,c in manifest['cohorts'].items():
 assert row_counts[name]==len(c['rows'])*c['trials_per_task']
 assert report['cohorts'][name]['graded']['assistant_response_tokens']['p50']==base['agents'][name]['cost']['tokens_p50'],name
output=root+'/diagnostics/2026.11.26/s0-full-cost-profile.json';StoragePath(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':output,'cohorts':report['cohorts'],'saved_rows':row_counts}),flush=True)
'''
compile(program,'s0-full-cost-profile','exec')
outer='import subprocess\nfrom marin.rl.skyrl import skyrl_cpu_command\nsubprocess.run(skyrl_cpu_command("python","-c",'+repr(program)+'),check=True)'
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','7200','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-cost-profile-20261126b','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(args).returncode)
