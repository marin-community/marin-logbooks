"""A12 exact A8a first18 arrays; A13 immediate duplicate of ON tool replay occurrences."""
from pathlib import Path
from collections import Counter
import hashlib,json
import pyarrow.parquet as pq
from experiments.post_training.identity_path_b.trace_schedule import compact_row
from experiments.post_training.identity_path_b.ablation_data import pack_prefix
B=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');P=Path(__file__).resolve().parent;D=Path('/mnt/storage/karan-scratch/genet-path-b');parent=D/'ablation-a8-r1/A8a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((parent/'exposure.json').read_bytes());schedule=[json.loads(x)for x in (parent/'occurrence-ledger.jsonl').read_bytes().splitlines()]
assert len(schedule)==7266 and sha(parent/'occurrence-ledger.jsonl')==old['ordered_occurrence_ledger_sha256']
assert sum(x['source']=='identity'and x['enable_thinking']for x in schedule)==179
prereg=B/'bfcl-noise-r1/A12-A13-science-and-analysis-preregistration-marten-r1.json';assert sha(prereg)=='d3c966d90391dd7e1fabb5e6e29ecdb3382f454c71fd3a4a312878d03cea8738'
census=json.loads((B/'r5-refusal-cap-r1/source-census-and-selection.json').read_bytes());need={x['row_id']for x in schedule};on={x['row_id']for x in schedule if x['source']=='identity'and x['enable_thinking']};rows={}
for variant in ['B-base','B-traces']:
 pool=Path(census['pools'][variant]['path']);assert sha(pool)==census['pools'][variant]['sha256']
 for line in pool.open('rb'):
  raw=json.loads(line)
  if raw['id']in need and (variant=='B-base'and raw['id']not in on or variant=='B-traces'and raw['id']in on):
   row=compact_row(raw);rows[row.row_id]=row
replay=D/'replay-admission-r4/candidate-replay.jsonl';assert sha(replay)=='affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a'
for line in replay.open('rb'):
 raw=json.loads(line)
 if raw['id']in need:row=compact_row(raw);rows[row.row_id]=row
assert set(rows)==need
TOOL_FAMILIES={'tool_first_call','tool_call_after_result','tool_reply'}
transformed=[];decisions=[]
for original in schedule:
 row=rows[original['row_id']];eligible=original['source']=='replay' and original['enable_thinking'] and row.family in TOOL_FAMILIES
 transformed.append(dict(original,parent_A8a_occurrence=original['occurrence'],copy_index=0))
 if eligible:transformed.append(dict(original,parent_A8a_occurrence=original['occurrence'],copy_index=1))
 decisions.append({'parent_A8a_occurrence':original['occurrence'],'row_id':row.row_id,'source_id':row.source_id,'source':original['source'],'enable_thinking':row.enable_thinking,'family':row.family,'immediate_extra_copy':eligible})
assert any(x['immediate_extra_copy']for x in decisions)
(P/'A13-duplication-decisions.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n'for x in decisions))
parent_records=pq.read_table(parent/'packed.parquet').to_pylist();reports={}
for label,horizon,stream in [('A12',18,schedule),('A13',36,transformed)]:
 out=D/'ablation-a12-a13-r1'/label
 report=pack_prefix(rows,stream,out,truncate=True,updates=horizon)
 consumed=[json.loads(x)for x in (out/'occurrence-ledger.jsonl').read_bytes().splitlines()];records=pq.read_table(out/'packed.parquet').to_pylist();seqs=[json.loads(x)for x in (out/'sequence-rows.jsonl').read_bytes().splitlines()]
 assert len(records)==horizon*8 and consumed==[dict(x,occurrence=i)for i,x in enumerate(stream[:len(consumed)])]
 if label=='A12':
  assert records==parent_records[:144]
  assert report['cumulative_batches']==old['cumulative_batches'][:18]
 else:
  # All copies remain byte-identical row references; identity decisions/order are never changed.
  assert all(x['row_id']==schedule[x['parent_A8a_occurrence']]['row_id'] and x['enable_thinking']==schedule[x['parent_A8a_occurrence']]['enable_thinking']for x in consumed)
  assert [x['parent_A8a_occurrence']for x in consumed if x['source']=='identity']==[x['occurrence']for x in schedule[:max(y['parent_A8a_occurrence']for y in consumed)+1]if x['source']=='identity']
 mass=[sum(sum(x['target_weights'][1:])for x in records[i:i+8])for i in range(0,len(records),8)];assert len(mass)==horizon and min(mass)>0
 identity_ids={x['row_id']for x in consumed if x['source']=='identity'};updates=[i//8+1 for i,s in enumerate(seqs)if any(r in identity_ids for r in s['ids'])]
 parent_counts=Counter(x['row_id']for x in schedule if x['source']=='replay');actual_counts=Counter(x['row_id']for x in consumed if x['source']=='replay')
 report.update(arm=label,parent_A8a_packed_sha256=sha(parent/'packed.parquet'),parent_A8a_exposure_sha256=sha(parent/'exposure.json'),parent_A8a_ledger_sha256=sha(parent/'occurrence-ledger.jsonl'),preregistration_sha256=sha(prereg),parent_occurrences=7266,transformed_occurrences=len(stream),consumed_occurrences=len(consumed),unconsumed_transformed_suffix=len(stream)-len(consumed),retained_identity_ON_occurrences=sum(x['source']=='identity'and x['enable_thinking']for x in consumed),batch_shifted_supervision_mass=mass,first_identity_update=min(updates),last_identity_update=max(updates),all_batches_nonempty=True,original_A8a_identity_decisions_unchanged=True,identity_source_repeat_cap_unchanged=True,replay_max_same_row_occurrences=max(actual_counts.values()),parent_replay_max_same_row_occurrences=max(parent_counts.values()),first18_arrays_exact_A8a=label=='A12',replay_weight=1.0,identity_weight=1.0,selector_tool_families=sorted(TOOL_FAMILIES),selector='source==replay AND enable_thinking==true AND family in exact3toolallowlist',eligible_parent_duplicate_occurrences=sum(x['immediate_extra_copy']for x in decisions)if label=='A13'else 0,consumed_extra_tool_copies=sum(x.get('copy_index')==1 for x in consumed),consumed_eligible_tool_occurrences=sum(x['source']=='replay'and x['enable_thinking']and rows[x['row_id']].family in TOOL_FAMILIES for x in consumed),copy_decisions_sha256=sha(P/'A13-duplication-decisions.jsonl')if label=='A13'else None,parent_pool_hashes={v:census['pools'][v]['sha256']for v in ['B-base','B-traces']},replay_pool_sha256=sha(replay),schedule_rule='exactA8afirst144arrays'if label=='A12'else'everymatchingA8areplayoccurrencegetsoneimmediateidenticalcopy; unitweight; maximal36; no refill/suffixextension/crop/targetchange')
 (out/'exposure.json').write_text(json.dumps(report,indent=2)+'\n');reports[label]=report
summary={'preregistration_sha256':sha(prereg),'A8a':old,'arms':reports,'remote_calls':0,'comparison_caveats':['A12dose/grouping first18exact, fresh18optimizertrajectory checked separately','A13 changes consumed identity/replay doses/grouping and replay row repeats; appearance doubling is not lossweight2 or constantidentitydose','Full original A8a179 identity ON decisions frozen, actual retained number depends on36batchcutoff']}
(P/'data-counts.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({a:{k:r[k]for k in ['row_occurrences','raw_supervised_tokens','final_text_tokens','reasoning_text_tokens','retained_identity_ON_occurrences','consumed_extra_tool_copies','padding_only_sequences','first_identity_update','last_identity_update','unconsumed_transformed_suffix']}for a,r in reports.items()},indent=2))
