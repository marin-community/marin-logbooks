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
report=dict(scope='UNTRUNCATED exactA0 occurrence prefix after A6replacement; schedule choice pending if >288sequences',parent_occurrences=len(schedule),required_unpadded_sequences=sequences,required_padded_sequences=((sequences+7)//8)*8,required_updates=(sequences+7)//8,fits36=sequences<=288,input_positions=position_count,raw_supervised_tokens=dict(raw),final_text_tokens=dict(final),reasoning_text_tokens=dict(reason),structural_target_tokens=dict(structural),ON_decisions=dict(Counter(d['decision'] for d in decisions)),trace_family_encounters=dict(trace_count),all_original_source_occurrences_replay_order_preserved=True,canonical_finals_and_GLMon_IDs_unchanged=True,model_GPU_remote_calls=0)
(HERE/'untruncated-A0-prefix-census.json').write_text(json.dumps(report,indent=2)+'\n');(HERE/'ON-decisions-untruncated.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in decisions));print(json.dumps(report,indent=2))
