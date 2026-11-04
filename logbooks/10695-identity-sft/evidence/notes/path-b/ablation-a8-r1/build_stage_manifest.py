from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
DATA=Path('/mnt/storage/karan-scratch/genet-path-b/ablation-a8-r1')
rows=[]
for arm in ['A8a','A8b']:
 for name in ['packed.parquet','exposure.json','occurrence-ledger.jsonl']:
  path=DATA/arm/name
  rows.append(dict(local=str(path),uri=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/a8-onshare-20261030-r1/inputs/{arm}/{name}',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(HERE/'stage-files.json').write_text(json.dumps(dict(files=rows,policy='CPU staging only, fresh14d immutable inputs; fullGET SHA/length verification; no overwrite or model calls'),indent=2)+'\n')
print(hashlib.sha256((HERE/'stage-files.json').read_bytes()).hexdigest(),sum(r['bytes'] for r in rows))
