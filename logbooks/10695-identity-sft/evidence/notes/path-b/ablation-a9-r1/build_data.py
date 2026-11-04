from pathlib import Path
import hashlib,json
from collections import Counter
import pyarrow.parquet as pq
from experiments.post_training.identity_path_b.trace_schedule import compact_row
from experiments.post_training.identity_path_b.ablation_data import pack_prefix
B=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');P=B/'ablation-a9-r1';P.mkdir(exist_ok=True)
D=Path('/mnt/storage/karan-scratch/genet-path-b');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ledger=D/'r5-weight1-r1/b-base/arm10/occurrence-ledger.jsonl';assert sha(ledger)=='0b46c96740a32ad420b3614259b77028871622f69434ef15682478a339714da0'
schedule=[json.loads(x) for x in ledger.read_bytes().splitlines()]
assert all(x['occurrence']==i for i,x in enumerate(schedule))
census=json.loads((B/'r5-refusal-cap-r1/source-census-and-selection.json').read_bytes())
base={};trace={};rows={}
for variant,destination in [('B-base',base),('B-traces',trace)]:
 path=Path(census['pools'][variant]['path']);assert sha(path)==census['pools'][variant]['sha256']
 for line in path.open('rb'):
  row=json.loads(line);destination[row['id']]=row
replay=D/'replay-admission-r4/candidate-replay.jsonl';assert sha(replay)=='affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a'
for line in replay.open('rb'):
 row=json.loads(line);rows[row['id']]=compact_row(row)
off={r['source_id']:r for r in base.values() if not r['enable_thinking']}
old=[json.loads(x) for x in (D/'ablation-a8-r1/A8a/occurrence-ledger.jsonl').read_bytes().splitlines()];assert len(old)==7266
oldchosen={json.loads(x)['occurrence'] for x in (B/'ablation-a8-r1/A8a-selected-ON-occurrences.jsonl').read_bytes().splitlines()};assert len(oldchosen)==179
eligible=[x for x in schedule[7266:] if x['source']=='identity' and x['enable_thinking'] and trace[x['row_id']].get('reasoning_text_tokens',0)>0]
ranked=sorted(eligible,key=lambda x:(hashlib.sha256(f"310:A8:{x['occurrence']}".encode()).hexdigest(),x['occurrence']))
chosen={x['occurrence'] for x in ranked[:len(ranked)//10]}|oldchosen
ranks=[dict(rank=i,occurrence=x['occurrence'],row_id=x['row_id'],source_id=x['source_id'],hash_input=f"310:A8:{x['occurrence']}",sha256=hashlib.sha256(f"310:A8:{x['occurrence']}".encode()).hexdigest(),selected=i<len(ranked)//10) for i,x in enumerate(ranked)]
(P/'entire-tail-eligible-ranked.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in ranks))
transformed=[];decisions=[]
for orig in schedule:
 occ=dict(orig)
 if occ['source']=='identity':
  if orig['enable_thinking'] and orig['occurrence'] in chosen:
   row=trace[orig['row_id']];assert row['enable_thinking'] and row.get('reasoning_text_tokens',0)>0;decision='GLM_ON'
  elif orig['enable_thinking']:
   row=off[orig['source_id']];occ.update(row_id=row['id'],enable_thinking=False);decision='same_source_OFF'
  else:row=base[orig['row_id']];decision='unchanged_OFF'
  compact=compact_row(row);rows[compact.row_id]=compact
  assert compact.dose.final_text==orig['canonical_final_text_tokens']
  if orig['enable_thinking']:decisions.append(dict(occurrence=orig['occurrence'],source_id=orig['source_id'],parent_row_id=orig['row_id'],new_row_id=compact.row_id,decision=decision,family=compact.family))
 transformed.append(occ)
assert transformed[:7266]==old
out=D/'ablation-a9-r1/A9';report=pack_prefix(rows,transformed,out,truncate=True,updates=72)
consumed=[json.loads(x) for x in (out/'occurrence-ledger.jsonl').read_bytes().splitlines()]
assert consumed==transformed[:len(consumed)] and len(consumed)>7266
record=pq.read_table(out/'packed.parquet').to_pylist();parent=pq.read_table(D/'ablation-a8-r1/A8a/packed.parquet').to_pylist()
assert len(record)==576 and record[:280]==parent[:280]
changed36=[i for i in range(280,288) if record[i]!=parent[i]]
mass=[sum(sum(x['target_weights'][1:]) for x in record[i:i+8]) for i in range(0,576,8)];assert len(mass)==72 and min(mass)>0
consumed_decisions=[x for x in decisions if x['occurrence']<len(consumed)]
(P/'all-ON-decisions.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in decisions))
report.update(arm='A9',parent_full_r5_ledger_sha256=sha(ledger),parent_A8a_ledger_sha256=sha(D/'ablation-a8-r1/A8a/occurrence-ledger.jsonl'),parent_identity_pool_sha256=census['pools']['B-base']['sha256'],trace_identity_pool_sha256=census['pools']['B-traces']['sha256'],parent_replay_pool_sha256=sha(replay),old_prefix_occurrences=7266,old_selected_trace_occurrences=179,entire_frozen_tail_eligible_ON=len(ranked),entire_tail_selected_trace_occurrences=len(ranked)//10,tail_ranked_sha256=sha(P/'entire-tail-eligible-ranked.jsonl'),all_ON_decisions_sha256=sha(P/'all-ON-decisions.jsonl'),retained_traced_ON_occurrences=sum(x['decision']=='GLM_ON' for x in consumed_decisions),eligible_consumed_traced_ON_occurrences=sum(x['source']=='identity' and x['enable_thinking'] and trace[x['row_id']].get('reasoning_text_tokens',0)>0 for x in schedule[:len(consumed)]),consumed_parent_occurrences=len(consumed),unconsumed_parent_occurrences=len(schedule)-len(consumed),first35_updates_arrays_exact_A8a=True,update36_changed_sequence_indices=changed36,update36_padding_fill_delta='A8a sequences0..280 unchanged; seven padding sequences281..287 filled by continuing original order; no first36 equality claim',batch_shifted_supervision_mass=mass,each_of72_batches_nonempty=True,trace_ON_family_occurrences=dict(Counter(x['family'] for x in consumed_decisions if x['decision']=='GLM_ON')),converted_ON_family_occurrences=dict(Counter(x['family'] for x in consumed_decisions if x['decision']=='same_source_OFF')),schedule_authority='Jerboa approved entire frozen tail; preserve all old7266 decisions/179 traces; seed310:A8:occurrence; fixed72 maximal prefix; no refill/rebalance/resample/cycle',subset_hash_input='UTF8(f"310:A8:{occurrence}")')
assert report['maximum_identity_source_occurrences_across_modes']<=3
report['consumed_traced_ON_share']=report['retained_traced_ON_occurrences']/report['eligible_consumed_traced_ON_occurrences']
(out/'exposure.json').write_text(json.dumps(report,indent=2)+'\n')
(P/'data-counts.json').write_text(json.dumps(dict(A9=report,A8a=json.loads((D/'ablation-a8-r1/A8a/exposure.json').read_bytes()),local_model_remote_calls=0),indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['row_occurrences','final_text_tokens','reasoning_text_tokens','raw_supervised_tokens','retained_traced_ON_occurrences','eligible_consumed_traced_ON_occurrences','consumed_traced_ON_share','entire_frozen_tail_eligible_ON','entire_tail_selected_trace_occurrences','padding_only_sequences','update36_changed_sequence_indices','maximum_identity_source_occurrences_across_modes']},indent=2))
