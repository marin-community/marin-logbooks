import os,subprocess
from pathlib import Path
inner = r'''
import hashlib,json,sys,shutil,subprocess,tarfile
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from cloud.iris.launch_config import load_launch_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rigging.redaction import redact_string
from rolloutengine.machines import _machine_spec
from rolloutengine.models import TASK_ROLLOUT_COLUMN,TaskRolloutSpec
from shellbox.backends.daytona.snapshot import snapshot_request_for_spec
from skyrl_train.dataset.tasks import _canonical_task_prompt
from taskcompendium.models import grader_environment
configure_coreweave_s3()
config=load_launch_config(Path(sys.argv[1]))
environments=OmegaConf.to_container(config.skyrl.environment.task_sessions,resolve=True)
pool='s3://marin-us-east-08a/marin/rl-data/task-selection/users/benfeuer/mochi-loop-guards/snowball-100k-agentic-ho-excluded/2026.11.25.1'
output='s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/prepared'
counts=Counter();families=Counter();requests={};roles=Counter();runtimes={};native_specs={}
with (StoragePath(pool)/'train.parquet').open('rb') as stream:
    for batch in pq.ParquetFile(stream).iter_batches(batch_size=128,columns=['family','task_spec']):
        for row in batch.to_pylist():
            families[row['family']]+=1
            if row['family']!='agentic':continue
            bound=_canonical_task_prompt(row,0,environment_configs=environments)
            spec=TaskRolloutSpec.model_validate_json(bound[TASK_ROLLOUT_COLUMN])
            for role,requirements,runtime in [('task',spec.task.environment_requirements,spec.task_machine),('verifier',grader_environment(spec.task.grader),spec.verifier_machine)]:
                if runtime is None:continue
                roles[(role,runtime.backend)]+=1
                if runtime.backend not in ('daytona','daytona-eval'):continue
                assert requirements is not None
                with _machine_spec(requirements,runtime) as machine:
                    request=snapshot_request_for_spec(machine)
                    native=request.as_dict()
                    if request.name not in native_specs:
                        if 'dockerfile' in native:
                            context=Path('/tmp/s2-required-contexts')/request.name
                            shutil.copytree(machine.source.context,context)
                            native['dockerfile']=str(context/Path(machine.source.dockerfile).relative_to(machine.source.context))
                            archive=context.with_suffix('.tar.gz')
                            with tarfile.open(archive,'w:gz') as bundle:bundle.add(context,arcname=request.name)
                            copy(str(archive),str(StoragePath(output)/'contexts'/archive.name))
                        native_specs[request.name]=native
                    payload=dict(native_specs[request.name])
                    if 'dockerfile' in payload:
                        payload['source_kind']='dockerfile'
                        payload['dockerfile_sha256']=hashlib.sha256(Path(payload.pop('dockerfile')).read_bytes()).hexdigest()
                        payload['context_archive']=str(StoragePath(output)/'contexts'/(request.name+'.tar.gz'))
                    else:
                        payload['source_kind']='registry_image'
                        assert '@sha256:' in payload['image']
                    requests[request.name]=payload
                    counts[request.name]+=1
                    runtimes[request.name]=runtime.model_dump(mode='json')
assert families['agentic']==5480 and sum(families.values())==105480
specs=sorted(requests.values(),key=lambda x:x['name'])
result={'gpu_submitted':False,'provider_mutations':False,'config_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),'pool':pool,'families':dict(families),'unique_snapshot_requests':len(specs),'machine_roles':[{'role':role,'backend':backend,'count':n} for (role,backend),n in roles.items()],'request_counts':dict(counts),'runtimes':runtimes}
(StoragePath(output)/'required-snapshots.json').write_text(json.dumps(specs,indent=2)+'\n')
(StoragePath(output)/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('request_counts','runtimes')}))
Path('/tmp/s2-required-specs.json').write_text(json.dumps(list(native_specs.values())))
base=[sys.executable,'-m','shellbox.backends.daytona.snapshot_cli','--api-key-env','DAYTONA_RL_API_KEY']
before=subprocess.run([*base,'inventory','--json'],capture_output=True,text=True,timeout=120)
if before.returncode:raise RuntimeError('RL snapshot inventory failed before ensure')
prior=json.loads(before.stdout)
prior_names={r['name'] for r in prior}
ensure=subprocess.run([*base,'ensure','--specs','/tmp/s2-required-specs.json'],capture_output=True,text=True,timeout=2700)
after=subprocess.run([*base,'inventory','--json'],capture_output=True,text=True,timeout=120)
if after.returncode:raise RuntimeError('RL snapshot inventory failed after ensure')
following={r['name']:r for r in json.loads(after.stdout)}
ready={name:following.get(name,{}).get('state') for name in native_specs}
preflight={'preflight_completed':True,'gpu_submitted':False,'model_trials':0,'snapshot_creation_attempted':True,'ensure_exit_code':ensure.returncode,'all_required_active':all(state=='active' for state in ready.values()),'required_states':ready,'required_present_before':sorted(set(native_specs)&prior_names),'provider_diagnostic':redact_string(ensure.stdout+'\n'+ensure.stderr)[-4000:]}
(StoragePath(output)/'snapshot-preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
print(json.dumps(preflight))
'''
outer = '\n'.join([
'import os,subprocess',
'from pathlib import Path',
'from rigging.secrets import require_secret',
'from marin.execution.lazy import materialized_config',
'from marin.rl.skyrl import skyrl_cpu_command',
'from rigging.filesystem.s3_compat import configure_coreweave_s3',
'from rigging.filesystem.storage_path import StoragePath',
'from experiments.post_training import rl_compose',
'from experiments.post_training.mochi_loop_guards.presets import S2_CONTROL',
"configure_coreweave_s3()",
"os.environ['MARIN_PREFIX']='s3://marin-us-east-08a/tmp/ttl=3d/marin'",
"rl_compose.SELECTION_ROOT='s3://marin-us-east-08a/marin/rl-data/task-selection'",
"spec=rl_compose.run_spec(S2_CONTROL,name=S2_CONTROL.name,model_name='mochi')",
"step=rl_compose.run_step(S2_CONTROL,spec)",
"source=materialized_config(step,os.environ['MARIN_PREFIX']).launch_config_yaml",
"Path('/tmp/s2-environment-launch.yaml').write_text(source)",
"StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/prepared/planned-native-launch.yaml').write_text(source)",
"os.environ['PYTHONPATH']=':'.join(str(Path.cwd()/p) for p in ('vendor/skyrl','vendor/skyrl/skyrl-train','vendor/skyrl/skyrl-gym'))",
f"subprocess.run(skyrl_cpu_command('python','-c',{inner!r},'/tmp/s2-environment-launch.yaml'),check=True)",
])
compile(inner,'s2-environment-census','exec')
compile(outer,'s2-environment-census-driver','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','16','--memory','64GB','--disk','128GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-s2-snapshot-preflight-20261125a','--secret','DAYTONA_RL_API_KEY','env:DAYTONA_RL_API_KEY','--','uv','run','--package','marin-core','python','-c',outer]
credential=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret','DAYTONA_RL_API_KEY'],capture_output=True,text=True)
if credential.returncode:raise RuntimeError('RL secret resolution failed')
os.environ['DAYTONA_RL_API_KEY']=credential.stdout.strip()
raise SystemExit(subprocess.run(args).returncode)
