import subprocess
import textwrap

program=textwrap.dedent('''
import hashlib,io,json
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import TASK_ROLLOUT_COLUMN,TaskRolloutSpec
configure_coreweave_s3()
ledger_uri='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/partial/tb2/repetition-ledger.json'
ledger=json.loads(StoragePath(ledger_uri).read_text())
if not ledger['frozen_seed_verified'] or not ledger['content_multisets_match']:raise ValueError('Unverified source identities')
failures=[row for row in ledger['trials'] if row['original_infrastructure_error']]
task_ids={row['task_id'] for row in failures}
if len(failures)!=33 or len(task_ids)!=18:raise ValueError('Recovery census differs from registered33trials/18tasks')
source='s3://marin-us-east-08a/marin/rl-data/task-selection/users/benfeuer/mochi-loop-guards/panel/2026.11.25.1/tb2.parquet'
payload=StoragePath(source).read_bytes();table=pq.read_table(io.BytesIO(payload));values=table[TASK_ROLLOUT_COLUMN].to_pylist()
indices=[i for i,value in enumerate(values) if TaskRolloutSpec.model_validate_json(value).task.id in task_ids]
selected=table.take(indices)
if len(selected)!=18:raise ValueError('Frozen task lookup incomplete')
rows=selected[TASK_ROLLOUT_COLUMN].to_pylist()
if rows!=[values[i] for i in indices]:raise ValueError('TaskRolloutSpecs changed')
root=StoragePath('s3://marin-us-east-08a/marin/rl-data/task-selection/users/benfeuer/mochi-loop-guards/s0-infra-recovery-tb2/2026.11.25.1')
root.mkdirs();buffer=io.BytesIO();pq.write_table(selected,buffer);data=buffer.getvalue()
if (root/'tb2.parquet').exists() and (root/'tb2.parquet').read_bytes()!=data:raise ValueError('Recovery selection already exists with different bytes')
(root/'tb2.parquet').write_bytes(data)
(root/'summary.json').write_text(json.dumps({'grader_kind_counts':{'train':{},'tb2':{}},'selection_version':'2026.11.25.1','tb2_rows':18},indent=2)+'\\n')
receipt={'original_ledger':ledger_uri,'original_source':source,'original_parquet_sha256':hashlib.sha256(payload).hexdigest(),'selection_sha256':hashlib.sha256(data).hexdigest(),'selected_task_ids':sorted(task_ids),'source_row_indices':indices,'selected_spec_sha256':[hashlib.sha256(value.encode()).hexdigest() for value in rows],'replayed_trials':54,'replaced_original_infrastructure_trials':33,'preserved_original_graded_trials':234,'extra_replayed_original_graded_trials_ignored':21,'repetitions_per_task':3,'rule':'one replay; replace only frozen original ungraded task/repetition identities; preserve all original graded outcomes','target_trials':failures}
(root/'recovery-receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps({key:value for key,value in receipt.items() if key not in ('target_trials','selected_spec_sha256','selected_task_ids','source_row_indices')}))
''')
compile(program,'publish-tb2-recovery-selection','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','4','--memory','16GB','--disk','64GB','--enable-extra-resources','--timeout','1800','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-tb2-recovery-selection-20261125a','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(args).returncode)
