"""Approved pre-outcome44source/440occurrence schedule and exhaustive packing fit."""
from pathlib import Path
from collections import Counter
from bisect import bisect_right
import hashlib,json
import pyarrow.parquet as pq
from experiments.post_training.identity_path_b.trace_schedule import compact_row,checked_replacement
from experiments.post_training.identity_path_b.ablation_data import pack_prefix
B=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');HERE=B/'ablation-a10-r1';D=Path('/mnt/storage/karan-scratch/genet-path-b');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
census=json.loads((B/'r5-refusal-cap-r1/source-census-and-selection.json').read_bytes());tracepath=Path(census['pools']['B-traces']['path']);basepath=Path(census['pools']['B-base']['path'])
assert sha(tracepath)==census['pools']['B-traces']['sha256'] and sha(basepath)==census['pools']['B-base']['sha256']
eligible=[]
for line in tracepath.open('rb'):
 row=json.loads(line)
 if row['enable_thinking'] and row.get('reasoning_text_tokens',0)>0:
  eligible.append(dict(source_id=row['source_id'],family=row['family'],row_id=row['id'],source_hash=hashlib.sha256(f"310:A10:{row['source_id']}".encode()).hexdigest()))
counts=Counter(r['family'] for r in eligible);assert len(eligible)==19847 and len(counts)==11
if min(counts.values())<4:raise ValueError('STOP: fewerthan4 eligible sources in family: '+str({f:n for f,n in counts.items() if n<4}))
selected=[]
for family in sorted(counts):
 ranked=sorted((r for r in eligible if r['family']==family),key=lambda r:(r['source_hash'],r['source_id']))
 for rank,row in enumerate(ranked[:4]):
  chosen=dict(row,family_rank=rank,ON_round=int(hashlib.sha256(f"310:A10:ON:{row['source_id']}".encode()).hexdigest(),16)%10)
  selected.append(chosen)
assert len(selected)==len({r['source_id'] for r in selected})==44
selection=dict(eligible_originals=19847,eligible_family_counts=dict(counts),source_hash_input='UTF8(f"310:A10:{source_id}")',ON_round_hash_input='UTF8(f"310:A10:ON:{source_id}")',selection='first4 per family by(hash,source_id)',selected=selected,authority_prebuild_binding_sha256=sha(B/'bfcl-noise-r1/A10-prebuild-data-binding-marten-r1.json'))
# Source IDs freeze before any length/fit calculation. No selection by token fit.
selectionpath=HERE/'44-source-selection-frozen.json';assert not selectionpath.exists();selectionpath.write_text(json.dumps(selection,indent=2)+'\n')
needed={r['source_id'] for r in selected};base={};trace={}
for path,into in [(basepath,base),(tracepath,trace)]:
 for line in path.open('rb'):
  row=json.loads(line)
  if row['source_id'] in needed:into[row['id']]=row
rows={};identity=[]
for roundindex in range(10):
 for spec in selected:
  on=roundindex==spec['ON_round'];rowid=spec['row_id'] if on else spec['row_id'].removesuffix('-on')+'-off'
  row=trace[rowid] if on else base[rowid]
  compact=checked_replacement(base[rowid],row) if on else compact_row(row);rows[rowid]=compact
  identity.append(dict(source='identity',source_id=row['source_id'],row_id=rowid,family=row['family'],enable_thinking=on,canonical_final_text_tokens=compact.dose.final_text,A10_round=roundindex,A10_source_rank=spec['family_rank']))
assert len(identity)==440 and sum(r['enable_thinking'] for r in identity)==44
parentledger=D/'r5-weight1-r1/b-base/arm10/occurrence-ledger.jsonl';assert sha(parentledger)=='0b46c96740a32ad420b3614259b77028871622f69434ef15682478a339714da0'
replayledger=[json.loads(x) for x in parentledger.read_bytes().splitlines() if json.loads(x)['source']=='replay'];needed_replay={r['row_id'] for r in replayledger}
replaypath=D/'replay-admission-r4/candidate-replay.jsonl';assert sha(replaypath)=='affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a'
for line in replaypath.open('rb'):
 row=json.loads(line)
 if row['id'] in needed_replay:rows[row['id']]=compact_row(row)
identity_lengths=[len(rows[r['row_id']].input_ids) for r in identity];replay_lengths=[len(rows[r['row_id']].input_ids) for r in replayledger]
assert max(identity_lengths+replay_lengths)<=8192
cumulative=[0]
for size in replay_lengths:cumulative.append(cumulative[-1]+size)
capacity=288*8192;upper=bisect_right(cumulative,capacity-sum(identity_lengths))-1;assert 0<upper<len(replayledger)

def merged(N):
 j=0
 for slot in range(N+1):
  while j<440 and (j+1)*N//441==slot:
   yield True,j;j+=1
  if slot<N:yield False,slot
 assert j==440

def packed_count(N):
 sequences=0;used=0
 for isidentity,index in merged(N):
  length=identity_lengths[index] if isidentity else replay_lengths[index]
  if used and used+length>8192:sequences+=1;used=0
  used+=length
 return sequences+bool(used)

fits=[dict(replay_prefix=N,sequences=packed_count(N)) for N in range(upper+1)]
N=max(x['replay_prefix'] for x in fits if x['sequences']<=288)
assert all(x['sequences']>288 for x in fits if x['replay_prefix']>N)
(HERE/'exhaustive-replay-prefix-fit.json').write_text(json.dumps(dict(frozen_selection_sha256=sha(selectionpath),capacity_token_bound=capacity,identity_input_tokens=sum(identity_lengths),replay_token_capacity_upper_N=upper,all_N_above_upper_excluded_by_sum_token_lower_bound=True,selected_largest_N=N,fit_results=fits),indent=2)+'\n')
schedule=[]
for isidentity,index in merged(N):
 occ=dict(identity[index]) if isidentity else dict(replayledger[index],parent_replay_occurrence=replayledger[index]['occurrence'])
 occ['occurrence']=len(schedule);schedule.append(occ)
output=D/'ablation-a10-r1/A10';report=pack_prefix(rows,schedule,output,truncate=False)
actual=[json.loads(x) for x in (output/'occurrence-ledger.jsonl').read_bytes().splitlines()];assert actual==schedule
assert len([r for r in actual if r['source']=='identity'])==440
assert [dict(r,occurrence=replayledger[i]['occurrence']) for i,r in enumerate([{k:v for k,v in r.items() if k!='parent_replay_occurrence'} for r in actual if r['source']=='replay'])]==replayledger[:N]
seqs=[json.loads(x) for x in (output/'sequence-rows.jsonl').read_bytes().splitlines()];idset={r['row_id'] for r in identity};onset={r['row_id'] for r in identity if r['enable_thinking']};timing=[]
for update in range(36):
 ids=[identifier for seq in seqs[update*8:(update+1)*8] for identifier in seq['ids']];timing.append(dict(completed_updates=update+1,identity_occurrences=sum(r in idset for r in ids),GLM_ON_occurrences=sum(r in onset for r in ids)))
active=[x['completed_updates'] for x in timing if x['identity_occurrences']];records=pq.read_table(output/'packed.parquet').to_pylist();mass=[sum(sum(r['target_weights'][1:]) for r in records[i:i+8]) for i in range(0,288,8)];assert len(records)==288 and min(mass)>0
assert report['maximum_identity_source_occurrences_across_modes']==10 and report['maximum_identity_rendered_row_occurrences']==9
report.update(arm='A10',selected_source_count=44,identity_occurrences=440,GLM_ON_occurrences=44,OFF_occurrences=396,source_repeat_authority='Owner A10explicit10/source supersedes prior<=3 ONLYforthisarm',source_selection_sha256=sha(selectionpath),exhaustive_fit_sha256=sha(HERE/'exhaustive-replay-prefix-fit.json'),replay_consumed_occurrences=N,replay_unconsumed_occurrences=len(replayledger)-N,parent_full_r5_ledger_sha256=sha(parentledger),parent_identity_pool_sha256=census['pools']['B-base']['sha256'],trace_identity_pool_sha256=census['pools']['B-traces']['sha256'],parent_replay_pool_sha256=sha(replaypath),identity_first_update=min(active),identity_last_update=max(active),identity_timing_by_update=timing,batch_shifted_supervision_mass=mass,all36_batches_nonempty_supervision=True,interleave='identity j before replay floor((j+1)*N/441); largest fittingN via exhaustive0..capacityupper; all440required',schedule_authority='Jerboa exactprebuildapproved44/4per11families x10,9OFF+1hashroundGLMON; unchanged replayprefix order;36updates; no refill/targetedit/newgeneration')
(output/'exposure.json').write_text(json.dumps(report,indent=2)+'\n')
(HERE/'data-counts.json').write_text(json.dumps(dict(A10=report,A8a=json.loads((D/'ablation-a8-r1/A8a/exposure.json').read_bytes()),model_remote_calls=0),indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['row_occurrences','replay_consumed_occurrences','identity_first_update','identity_last_update','final_text_tokens','reasoning_text_tokens','raw_supervised_tokens','padding_only_sequences','maximum_identity_source_occurrences_across_modes']},indent=2))
