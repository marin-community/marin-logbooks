import json, subprocess
from pathlib import Path
payload={p.name:json.loads(p.read_text()) for p in Path('/tmp/mochi-loop-guards/user-overrides-20261126').glob('*.json')}
hf=json.loads(Path('/tmp/mochi-loop-guards/hf-conservation-inventory.json').read_text())
program='payload='+repr(payload)+'\nhf='+repr(hf)+'\n'+r'''
import json
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
configure_coreweave_s3()
root=StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards')
for name,receipt in payload.items():
 (root/'decisions'/name).write_text(json.dumps(receipt,indent=2)+'\n')
source=StoragePath(payload['checkpoint-conservation-user-override.json']['source'])
(source/'.conservation.json').write_text(json.dumps(payload['checkpoint-conservation-user-override.json'],indent=2)+'\n')
(root/'conservation'/'sft-400407-HF-inventory.json').write_text(json.dumps(hf,indent=2)+'\n')
entries=[]
for file in (source/'*').glob():
 if file.isfile():
  entries.append({'name':file.name,'size':file.size()})
(root/'conservation'/'sft-400407-regional-inventory.json').write_text(json.dumps(entries,indent=2,default=str)+'\n')
for entry in entries:
 if 'manifest' in entry['name'].lower() or 'sha' in entry['name'].lower():
  if entry['size'] < 10_000_000:
   text=(source/entry['name']).read_text()
   (root/'conservation'/entry['name']).write_text(text)
   print(entry['name'],text[:10000],flush=True)
print(json.dumps({'receipts_published':list(payload),'permanent_marker':str(source/'.conservation.json'),'regional_inventory_count':len(entries),'inventory':str(root/'conservation'/'sft-400407-regional-inventory.json')}),flush=True)
'''
compile(program,'regional-user-overrides','exec')
args=['uv','run','--no-sync','iris','--cluster','cw-us-east-08a','job','run','--no-wait','--priority','production','--cpu','8','--memory','32GB','--disk','64GB','--enable-extra-resources','--timeout','3600','--sync-package','marin-core','--job-name','mochi-loop-guards-user-overrides-20261126a','--','uv','run','--package','marin-core','python','-c',program]
raise SystemExit(subprocess.run(args).returncode)
