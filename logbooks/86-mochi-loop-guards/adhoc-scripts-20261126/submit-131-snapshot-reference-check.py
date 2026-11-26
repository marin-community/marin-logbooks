import subprocess
program=r'''
import json
from collections import Counter,defaultdict
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
configure_coreweave_s3()
root=StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-exp4-dspark/long-loop/mochi-exp4-original-rl-pools/2026.11.25.131')
fields=defaultdict(Counter);files=[];matches=0

def memory_fields(value,path=''):
    if isinstance(value,dict):
        for key,item in value.items():
            location=path+'.'+key
            if ('memory' in key or key in ('cpus','storage_mb')) and (item is None or isinstance(item,(int,float))):yield location,item
            elif isinstance(item,(dict,list)):yield from memory_fields(item,location)
    elif isinstance(value,list):
        for index,item in enumerate(value):yield from memory_fields(item,path+'.'+str(index))

for index in range(8):
    filename=f'source-{index:03}/train.parquet';count=0;selected=0
    with (root/filename).open('rb') as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32,columns=['task_spec']):
            for raw in batch.column(0).to_pylist():
                count+=1
                if 'long-loop-smartie-swesmith' not in raw:continue
                task=json.loads(raw)
                if task['source']['dataset']!='long-loop-smartie-swesmith':continue
                selected+=1;matches+=1
                for path,value in memory_fields(task):fields[path][json.dumps(value)]+=1
    files.append({'file':filename,'rows':count,'swesmith_rows':selected})
    print(json.dumps(files[-1]),flush=True)
result={'gpu_submitted':False,'provider_mutations':False,'source':'active .131 actual input parquet rows','files':files,'swesmith_rows':matches,'resource_binding_fields':{path:dict(counts) for path,counts in fields.items()}}
StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards/diagnostics/2026.11.25/s2-environments/131-resource-reference-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
compile(program,'snapshot-reference-131','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','4','--memory','16GB','--disk','64GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-snapshot-ref-131-20261125a','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(args).returncode)
