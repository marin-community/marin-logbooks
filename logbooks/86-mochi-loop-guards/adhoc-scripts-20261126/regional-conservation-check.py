import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

configure_coreweave_s3()
root=StoragePath('s3://marin-us-east-08a/marin/users/benfeuer/mochi-loop-guards')
source=StoragePath('s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1')
hf=json.loads((root/'conservation'/'sft-400407-HF-inventory.json').read_text())
manifest=json.loads((source/'.marinskyrl-model-manifest.json').read_text())
fs,prefix=filesystem_for(str(source))
listing={entry['name'].removeprefix(prefix+'/'):entry for entry in fs.ls(prefix,detail=True)}
manifest_files={r['path']:r for r in manifest['files']}
results=[]
for item in hf['siblings']:
 name=item['path'];registered=manifest_files.get(name)
 if registered is None:
  results.append({'path':name,'HF_repository_metadata_only':True});continue
 assert name in listing,name
 assert listing[name]['size']==registered['size']==item['size'],name
 if item['lfs']:
  assert item['lfs']['sha256']==registered['sha256'],name
 else:
  assert hashlib.sha256((source/name).read_bytes()).hexdigest()==registered['sha256'],name
 results.append({'path':name,'size':item['size'],'sha256':registered['sha256'],'size_match':True,'hash_match':True,'hash_basis':'retained verified-copy/source manifest and HF LFS SHA256' if name.endswith('.safetensors') else 'regional content read SHA256','regional_ETag':listing[name].get('ETag')})
assert sum(r['path'].endswith('.safetensors') for r in results)==193
index=json.loads((source/'model.safetensors.index.json').read_text())
assert set(index['weight_map'].values())=={r['path'] for r in results if r['path'].endswith('.safetensors')}
receipt={'complete':True,'HF_private':hf['private'],'HF_commit':hf['sha'],'source':str(source),'regional_manifest_identity':manifest['identity'],'common_files_verified':sum('hash_match' in r for r in results),'weight_shards_verified':193,'all_index_shards_present':True,'HF_repository_only_files':[r['path'] for r in results if r.get('HF_repository_metadata_only')],'source_weights_rehashed_this_run':False,'hash_basis':'Existing regional verified-copy SHA256 inventory compared with HF LFS SHA256; current regional file sizes verified. Small model files rehashed directly.','files':results}
(root/'conservation'/'sft-400407-verification-receipt.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)
