"""Build A0/A1/A2/A3 from the admitted r5 source bytes; no remote/model calls."""
from pathlib import Path
import hashlib,json
import pyarrow.parquet as pq
from experiments.post_training.identity_path_b.ablation_data import ordered_prefix,off_identity_occurrences,pack_prefix
from experiments.post_training.identity_path_b.trace_schedule import compact_row
B=Path('/mnt/storage/karan-scratch/genet-path-b')
PARENT=B/'r5-weight1-r1/b-base/arm10'
OUT=B/'bfcl-ablation-r1'
NOTES=Path(__file__).resolve().parent
IDENTITY=B/'r5-refusal-cap-r1/b-base-identity.jsonl'
REPLAY=B/'replay-admission-r4/candidate-replay.jsonl'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 OUT.mkdir(exist_ok=False)
 rows={}
 for path in [IDENTITY,REPLAY]:
  for line in path.open('rb'):
   r=json.loads(line);assert r['id'] not in rows;rows[r['id']]=compact_row(r)
 prefix=ordered_prefix(PARENT)
 allledger=[json.loads(line) for line in (PARENT/'occurrence-ledger.jsonl').read_bytes().splitlines()]
 replay=[r for r in allledger if r['source']=='replay']
 reports={}
 for arm,schedule,truncate in [('A0',prefix,False),('A1',prefix,False),('A2',replay,True),('A3',off_identity_occurrences(prefix,rows),False)]:
  report=pack_prefix(rows,schedule,OUT/arm,truncate=truncate)
  report['arm']=arm
  report['parent_identity_pool_sha256']=sha(IDENTITY);report['parent_replay_pool_sha256']=sha(REPLAY)
  report['parent_ledger_sha256']=sha(PARENT/'occurrence-ledger.jsonl')
  report['parent_packed_sha256']=sha(PARENT/'packed.parquet')
  (OUT/arm/'exposure.json').write_text(json.dumps(report,indent=2)+'\n');reports[arm]=report
 a=pq.read_table(PARENT/'packed.parquet').slice(0,288).to_pylist();b=pq.read_table(OUT/'A0/packed.parquet').to_pylist()
 assert a==b
 assert (OUT/'A0/packed.parquet').read_bytes()==(OUT/'A1/packed.parquet').read_bytes()
 assert reports['A3']['final_text_tokens']==reports['A0']['final_text_tokens']
 ar=[r for r in prefix if r['source']=='replay'];br=[json.loads(l) for l in (OUT/'A3/occurrence-ledger.jsonl').read_bytes().splitlines() if json.loads(l)['source']=='replay']
 assert [dict(r,occurrence=0) for r in ar]==[dict(r,occurrence=0) for r in br]
 summary={k:{f:v[f] for f in ['updates','row_occurrences','padding_only_sequences','raw_supervised_tokens','final_text_tokens','reasoning_text_tokens','mode_final_text_tokens','realized_identity_supervised_share','structural_target_tokens']} for k,v in reports.items()}
 (NOTES/'data-counts-A0-A3.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
