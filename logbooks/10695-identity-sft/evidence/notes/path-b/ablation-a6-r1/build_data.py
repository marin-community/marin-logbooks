from pathlib import Path
from collections import Counter
import hashlib,json
from experiments.post_training.identity_path_b.trace_schedule import compact_row,checked_replacement

HERE=Path(__file__).resolve().parent;B=HERE.parent;DATA=Path('/mnt/storage/karan-scratch/genet-path-b')
parent=DATA/'bfcl-ablation-r1/A0';schedule=list(map(json.loads,(parent/'occurrence-ledger.jsonl').read_bytes().splitlines()))
needed={row['source_id'] for row in schedule if row['source']=='identity'};replay_ids={row['row_id'] for row in schedule if row['source']=='replay'}
census=json.loads((B/'r5-refusal-cap-r1/source-census-and-selection.json').read_text());rows={};base={};trace={}
for variant,destination in [('B-base',base),('B-traces',trace)]:
 p=Path(census['pools'][variant]['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==census['pools'][variant]['sha256']
 for line in p.open('rb'):
  row=json.loads(line)
  if row['source_id'] in needed:destination[row['id']]=row
replay=DATA/'replay-admission-r4/candidate-replay.jsonl';assert hashlib.sha256(replay.read_bytes()).hexdigest()=='affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a'
for line in replay.open('rb'):
 row=json.loads(line)
 if row['id'] in replay_ids:rows[row['id']]=compact_row(row)
off={row['source_id']:row for row in base.values() if not row['enable_thinking']};new=[];decisions=[];trace_count=Counter();raw=Counter();final=Counter();reason=Counter();structural=Counter();position_count=0;used=0;sequences=0
for original in schedule:
 occ=dict(original)
 if original['source']=='identity':
  row=base[original['row_id']]
  if original['enable_thinking']:
   replacement=trace[original['row_id']]
   if compact_row(replacement).dose.reasoning_text:
    compact=checked_replacement(row,replacement);decision='existing_QCpassed_GLM_ON';trace_count[row['family']]+=1
   else:
    row=off[row['source_id']];compact=compact_row(row);occ.update(row_id=row['id'],enable_thinking=False);decision='same_source_OFF_fallback'
   decisions.append(dict(occurrence=original['occurrence'],source_id=original['source_id'],family=compact.family,decision=decision,original_row_id=original['row_id'],new_row_id=compact.row_id))
  else:compact=compact_row(row)
  rows[compact.row_id]=compact
 else:compact=rows[original['row_id']]
 size=len(compact.input_ids)
 if used and used+size>8192:sequences+=1;used=0
 used+=size;position_count+=size;new.append(occ)
 source=occ['source'];raw[source]+=compact.dose.raw;final[source]+=compact.dose.final_text;reason[source]+=compact.dose.reasoning_text;structural.update({f'{source}/{k}':v for k,v in compact.dose.structural.items()})
if used:sequences+=1

from experiments.post_training.identity_path_b.ablation_data import pack_prefix
output=DATA/'ablation-a6-r1/A6'
report=pack_prefix(rows,new,output,truncate=True)
report.update(arm='A6', parent_ledger_sha256=hashlib.sha256((parent/'occurrence-ledger.jsonl').read_bytes()).hexdigest(), parent_packed_sha256=hashlib.sha256((parent/'packed.parquet').read_bytes()).hexdigest(), parent_identity_pool_sha256=census['pools']['B-base']['sha256'], trace_identity_pool_sha256=census['pools']['B-traces']['sha256'], parent_replay_pool_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(), schedule_authority='Jerboa fixed36 deterministic maximal transformed prefix; no37/no refill', transformation='QCpassed existing GLM ON else exact same-source OFF; canonical finals unchanged')
selected=list(map(json.loads,(output/'occurrence-ledger.jsonl').read_bytes().splitlines()))
assert [dict(r,occurrence=0) for r in selected]==[dict(r,occurrence=0) for r in new[:len(selected)]]
report['consumed_parent_occurrences']=len(selected)
report['unconsumed_parent_occurrences']=len(new)-len(selected)
counts=Counter();families=Counter()
for original, occurrence in zip(schedule,selected,strict=False):
 if original['source']=='identity' and original['enable_thinking']:
  decision='GLM_ON' if occurrence['enable_thinking'] else 'OFF_fallback'
  counts[decision]+=1;families[rows[occurrence['row_id']].family+'/'+decision]+=1
report['original_ON_decisions']=dict(counts);report['family_ON_decisions']=dict(families)
(output/'exposure.json').write_text(json.dumps(report,indent=2)+'\n')
(HERE/'data-counts.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['raw_supervised_tokens','final_text_tokens','reasoning_text_tokens','row_occurrences','unconsumed_parent_occurrences','original_ON_decisions']},indent=2))
