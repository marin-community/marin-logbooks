"""Submit the archived, reviewed full Trainer check exactly once."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path('/private/tmp')
command_path = root/'native-jax-query-scale-15b-trainer-v2-command.json'
c = json.loads(command_path.read_text())
prepared = json.loads((root/'native-jax-query-scale-15b-trainer-v2-prepared.json').read_text())
assert hashlib.sha256(command_path.read_bytes()).hexdigest() == prepared['artifact_sha256']['command']
for name, path in prepared['artifacts'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == prepared['artifact_sha256'][name]
assert hashlib.sha256(Path(c['runner']).read_bytes()).hexdigest() == c['runner_sha256']
assert hashlib.sha256(c['argv'][-1].encode()).hexdigest() == c['compressed_launcher_sha256']
config = json.loads(Path(c['job_config_file']).read_text())
for path, sha in config['source_file_sha256'].items():
    assert hashlib.sha256((Path(c['working_directory'])/path).read_bytes()).hexdigest() == sha
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['working_directory'],text=True).strip() == c['source_pin']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=c['working_directory'],text=True).strip()
review = json.loads((root/'native-jax-query-scale-15b-trainer-v2-independent-review.json').read_text())
assert not review['source_blockers'] and review['final_v2_all_artifact_hashes_verified']
envelope = json.loads((root/'native-jax-query-scale-15b-trainer-v2-env-envelope-validation.json').read_text())
assert envelope['all_bounds_pass'] and envelope['no_forwarded_secrets']
archive = json.loads((root/'native-jax-durable-published-proof.json').read_text())
assert archive['archive_commit'] == '61403c428985aa1aa63ff35a779eb4d5ac40de4e'
assert archive['remote_notebook_manifest_v2_runner_and_command_bytes_exact']
assert archive['remote_automatic_workflows_absent']
now = datetime.datetime.now(datetime.timezone.utc)
absence = json.loads((root/'native-jax-query-scale-15b-trainer-v2-absence-proof.json').read_text())
assert len(absence) == 2 and all(row['confirmed_absent'] for row in absence)
assert all((now-datetime.datetime.fromisoformat(row['checked_utc'])).total_seconds()<600 for row in absence)
job_id = '/dlwh/' + c['job_name']
state_path = root/'marin-native-jax-query-scale-15b-trainer-v2-monitor.json'
state = {'job_id':job_id,'config':'lib/iris/config/marin.yaml','source':c['source_pin'],'status':'SUBMITTING','submit_started_utc':now.isoformat(),'submitted_utc':None,'exact_submit_command':'python3 /private/tmp/submit_native_jax_query_scale_15b_trainer_v2.py','frozen_command':str(command_path),'research_archive_commit':archive['archive_commit'],'last_observation_utc':None,'initial_delay_seconds':120,'later_observation_spacing_seconds':570,'backup_uri':c['backup_uri'],'restart_count':0,'no_blind_retry':True,'submission_log':str(root/'native-jax-query-scale-15b-trainer-v2-submission.log')}
with (root/'native-jax-query-scale-15b-trainer-v2-submission-started.json').open('x') as out:
    json.dump(state,out,indent=2)
state_path.write_text(json.dumps(state,indent=2)+'\n')
env = dict(os.environ,UV_PROJECT_ENVIRONMENT='/Users/dlwh/.codex/worktrees/73be/marin/.venv',UV_CACHE_DIR='/private/tmp/marin-native-jax-uv-cache')
with Path(state['submission_log']).open('w') as out:
    result = subprocess.run(c['argv'],cwd=c['working_directory'],env=env,stdout=out,stderr=subprocess.STDOUT)
state['submit_finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
state['submission_returncode'] = result.returncode
state['status'] = 'SUBMITTED_AWAITING_INITIAL_CHECK' if result.returncode == 0 else 'SUBMISSION_OUTCOME_UNKNOWN_INSPECT_SAME_JOB'
if result.returncode == 0:
    state['submitted_utc'] = state['submit_finished_utc']
state_path.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
raise SystemExit(result.returncode)
