from pathlib import Path
import subprocess
import textwrap

trajectory_root=Path('/tmp/mochi-loop-guards/s0-v9-trajectory-root.txt').read_text()
program=textwrap.dedent('''
import gzip,hashlib,io,json,zipfile
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.mismatch_probe.protocol import request_seed
from vendor.skyrl.scripts.loop_audit import read_rows,is_infrastructure_error
configure_coreweave_s3()
trajectory_root=TRAJECTORY_ROOT
source='s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/mochi-loop-guards/s0-baseline-f28d12bf4c-in-af5ab9b8/2026.11.25.9/exports/dumped_evals/a54355029268/eval_only/DCAgent2_terminal_bench_2.jsonl'
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
rows=list(read_rows(source));task_by_uid={row['uid']:row['task_id'] for row in rows}
def key(uid,tokens,messages,status,error):return (uid,digest(tokens),digest(messages),status,error)
raw_groups=defaultdict(list)
for row in rows:
    raw_groups[key(row['uid'],row['response_token_ids'],row.get('messages'),(row.get('verification') or {}).get('status'),row.get('exception_type'))].append(row)
def extract(path):
    result=[]
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        for name in archive.namelist():
            if not name.startswith('records/') or not name.endswith('.json.gz'):continue
            record=json.loads(gzip.decompress(archive.read(name)))
            extras=record['trajectory']['environment_extras']
            if extras.get('data_source')!='DCAgent2/terminal_bench_2':continue
            instance=record['trajectory']['instance_id'];uid=instance.rsplit('-',1)[-1]
            response=record['response'];verification=record.get('verification_result') or {}
            identity=key(uid,response['token_ids'],response.get('messages'),verification.get('status'),record['disposition'].get('exception_type'))
            result.append((identity,{'record_id':record['record_id'],'archive':str(path),'instance_id':instance,'uid':uid,'task_id':task_by_uid[uid],'repetition_id':record['trajectory']['repetition_id'],'sampling':record['provenance']['sampling'],'verification_status':verification.get('status'),'exception_type':record['disposition'].get('exception_type')}))
    return result
paths=sorted((StoragePath(trajectory_root)/'schema_v7/archives/phase=eval/step=eval_only/*.zip').glob(),key=str)
retained=defaultdict(list)
with ThreadPoolExecutor(max_workers=8) as pool:
    for records in pool.map(extract,paths):
        for identity,metadata in records:retained[identity].append(metadata)
if {identity:len(items) for identity,items in retained.items()}!={identity:len(items) for identity,items in raw_groups.items()}:
    raise ValueError('Retained/raw TB2 scientific-content multisets differ; do not infer repetition from completion ordinal')
ledger=[]
for identity,metadata in retained.items():
    originals=raw_groups[identity]
    for item in metadata:
        expected=request_seed(20261125,item['task_id'],item['repetition_id'])
        if item['sampling'].get('seed')!=expected:raise ValueError('Recorded trial seed differs from frozen seed contract')
        item['scientifically_equivalent_dump_ordinals']=sorted(row['ordinal'] for row in originals)
        item['join_key']=list(identity)
        item['original_infrastructure_error']=is_infrastructure_error(originals[0])
        ledger.append(item)
counts=Counter((item['task_id'],item['repetition_id']) for item in ledger)
if len(ledger)!=267 or len(counts)!=267 or set(counts.values())!={1}:raise ValueError('Retained repetitions are incomplete or duplicated')
receipt={'source':source,'trajectory_root':trajectory_root,'trial_count':len(ledger),'infrastructure_trials':sum(item['original_infrastructure_error'] for item in ledger),'frozen_seed_verified':True,'content_multisets_match':True,'trials':ledger}
StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/audits/s0-2026.11.25.9/partial/tb2/repetition-ledger.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps({key:value for key,value in receipt.items() if key!='trials'}))
''').replace('trajectory_root=TRAJECTORY_ROOT','trajectory_root='+repr(trajectory_root))
compile(program,'tb2-repetition-ledger','exec')
outer='from marin.rl.skyrl import skyrl_cpu_command; import subprocess; subprocess.run(skyrl_cpu_command("python","-c",'+repr(program)+'),check=True)'
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','32GB','--disk','64GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-s0-tb2-identities-20261125a','--','uv','run','--package','marin-core','python','-c',outer]
raise SystemExit(subprocess.run(args).returncode)
