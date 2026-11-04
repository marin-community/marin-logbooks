"""A8 nested ON occurrence subsets, preserving the exact consumed A6 ledger."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from experiments.post_training.identity_path_b.trace_schedule import compact_row
from experiments.post_training.identity_path_b.ablation_data import pack_prefix

HERE=Path(__file__).resolve().parent
B=HERE.parent
DATA=Path('/mnt/storage/karan-scratch/genet-path-b')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=DATA/'ablation-a6-r1/A6'
parent_report=json.loads((parent/'exposure.json').read_text())
schedule=[json.loads(line) for line in (parent/'occurrence-ledger.jsonl').read_bytes().splitlines()]
assert len(schedule)==7266 and parent_report['original_ON_decisions']['GLM_ON']==1791
assert sha(parent/'occurrence-ledger.jsonl')==parent_report['ordered_occurrence_ledger_sha256']
traced=[r for r in schedule if r['source']=='identity' and r['enable_thinking']]
assert len(traced)==1791 and len({r['occurrence'] for r in traced})==1791
ranked=sorted(traced,key=lambda r:(hashlib.sha256(f"310:A8:{r['occurrence']}".encode('utf-8')).hexdigest(),r['occurrence']))
rank_records=[dict(rank=i,occurrence=r['occurrence'],source_id=r['source_id'],row_id=r['row_id'],hash_input=f"310:A8:{r['occurrence']}",sha256=hashlib.sha256(f"310:A8:{r['occurrence']}".encode()).hexdigest()) for i,r in enumerate(ranked)]
(HERE/'eligible-traced-occurrences-seed310.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rank_records))
needed={r['source_id'] for r in schedule if r['source']=='identity'}
replay_ids={r['row_id'] for r in schedule if r['source']=='replay'}
census=json.loads((B/'r5-refusal-cap-r1/source-census-and-selection.json').read_text())
rows={};base={};trace={}
for variant,destination in [('B-base',base),('B-traces',trace)]:
 path=Path(census['pools'][variant]['path']);assert sha(path)==census['pools'][variant]['sha256']
 for line in path.open('rb'):
  row=json.loads(line)
  if row['source_id'] in needed:destination[row['id']]=row
replay=DATA/'replay-admission-r4/candidate-replay.jsonl';assert sha(replay)=='affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a'
for line in replay.open('rb'):
 row=json.loads(line)
 if row['id'] in replay_ids:rows[row['id']]=compact_row(row)
off={r['source_id']:r for r in base.values() if not r['enable_thinking']}
reports={}
for arm,count in [('A8a',179),('A8b',537)]:
 chosen={r['occurrence'] for r in ranked[:count]}
 selected_records=[r for r in rank_records if r['occurrence'] in chosen]
 (HERE/f'{arm}-selected-ON-occurrences.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected_records))
 transformed=[];decisions=[]
 for original in schedule:
  occ=dict(original)
  if occ['source']=='identity':
   if original['enable_thinking'] and original['occurrence'] in chosen:
    row=trace[original['row_id']];decision='same_A6_GLM_ON'
   elif original['enable_thinking']:
    row=off[original['source_id']];occ.update(row_id=row['id'],enable_thinking=False);decision='unselected_same_source_A3_OFF'
   else:
    row=base[original['row_id']];decision='unchanged_A6_OFF'
   compact=compact_row(row);rows[compact.row_id]=compact
   assert compact.dose.final_text==original['canonical_final_text_tokens']
   if original['enable_thinking']:decisions.append(dict(occurrence=original['occurrence'],source_id=original['source_id'],original_row_id=original['row_id'],new_row_id=compact.row_id,decision=decision,family=compact.family))
  transformed.append(occ)
 output=DATA/'ablation-a8-r1'/arm
 report=pack_prefix(rows,transformed,output,truncate=False)
 selected=[json.loads(x) for x in (output/'occurrence-ledger.jsonl').read_bytes().splitlines()]
 assert selected==transformed and len(selected)==7266
 assert [r for r in selected if r['source']=='replay']==[r for r in schedule if r['source']=='replay']
 assert report['final_text_tokens']==parent_report['final_text_tokens']
 assert report['raw_supervised_tokens']['replay']==parent_report['raw_supervised_tokens']['replay']
 assert report['reasoning_text_tokens']['replay']==parent_report['reasoning_text_tokens']['replay']
 records=pq.read_table(output/'packed.parquet').to_pylist()
 batch_mass=[sum(sum(r['target_weights'][1:]) for r in records[i:i+8]) for i in range(0,288,8)]
 assert len(records)==288 and len(batch_mass)==36 and min(batch_mass)>0
 assert len([r for r in selected if r['source']=='identity' and r['enable_thinking']])==count
 (HERE/f'{arm}-ON-decisions.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in decisions))
 report.update(arm=arm,parent_A6_ledger_sha256=sha(parent/'occurrence-ledger.jsonl'),parent_A6_exposure_sha256=sha(parent/'exposure.json'),parent_A6_packed_sha256=sha(parent/'packed.parquet'),parent_identity_pool_sha256=census['pools']['B-base']['sha256'],trace_identity_pool_sha256=census['pools']['B-traces']['sha256'],parent_replay_pool_sha256=sha(replay),selection_unit='consumed_A6_traced_ON_occurrence',subset_hash_input='UTF8(f"310:A8:{occurrence}")',subset_order='(SHA256 hex, parent integer occurrence)',eligible_traced_ON_occurrences=1791,retained_traced_ON_occurrences=count,retained_ON_fraction=count/1791,eligible_ranked_list_sha256=sha(HERE/'eligible-traced-occurrences-seed310.jsonl'),selected_ON_list_sha256=sha(HERE/f'{arm}-selected-ON-occurrences.jsonl'),ON_decisions_sha256=sha(HERE/f'{arm}-ON-decisions.jsonl'),schedule_authority='Exact consumed A6 7266 occurrences; no old175 suffix, truncation, refill, resampling or target edit; padding completes288 sequences',consumed_parent_occurrences=7266,unconsumed_parent_occurrences=0,old_A6_suffix_occurrences_excluded=175,batch_shifted_supervision_mass=batch_mass,each_of36_batches_has_nonempty_supervision=True,trace_ON_family_occurrences=dict(Counter(r['family'] for r in decisions if r['decision']=='same_A6_GLM_ON')),converted_ON_family_occurrences=dict(Counter(r['family'] for r in decisions if r['decision']=='unselected_same_source_A3_OFF')))
 (output/'exposure.json').write_text(json.dumps(report,indent=2)+'\n')
 reports[arm]=report
assert {r['occurrence'] for r in ranked[:179]} <= {r['occurrence'] for r in ranked[:537]}
summary=dict(parent_consumed_A6_ledger_sha256=sha(parent/'occurrence-ledger.jsonl'),eligible_ranked_list_sha256=sha(HERE/'eligible-traced-occurrences-seed310.jsonl'),subset_scope='179/1791 and537/1791, nested occurrence-level selection before GPU outcomes',A6=parent_report['cumulative_batches'][-1],A3=json.loads((DATA/'bfcl-ablation-r1/A3/exposure.json').read_text())['cumulative_batches'][-1],arms=reports,remote_calls_by_genet=0)
(HERE/'data-counts.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({a:{k:r[k] for k in ['row_occurrences','retained_traced_ON_occurrences','raw_supervised_tokens','final_text_tokens','reasoning_text_tokens','padding_only_sequences','batch_shifted_supervision_mass']} for a,r in reports.items()},indent=2))
