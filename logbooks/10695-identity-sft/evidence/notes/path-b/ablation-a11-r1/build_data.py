from pathlib import Path
import hashlib,json
from collections import Counter
import pyarrow.parquet as pq
from experiments.post_training.identity_path_b.trace_schedule import compact_row
from experiments.post_training.identity_path_b.ablation_data import pack_prefix,off_identity_occurrences
B=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b');P=B/'ablation-a11-r1';P.mkdir(exist_ok=True)
D=Path('/mnt/storage/karan-scratch/genet-path-b');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ledger=D/'r5-weight1-r1/b-base/arm10/occurrence-ledger.jsonl';assert sha(ledger)=='0b46c96740a32ad420b3614259b77028871622f69434ef15682478a339714da0'
schedule=[json.loads(x) for x in ledger.read_bytes().splitlines()];assert all(x['occurrence']==i for i,x in enumerate(schedule))
census=json.loads((B/'r5-refusal-cap-r1/source-census-and-selection.json').read_bytes());path=Path(census['pools']['B-base']['path']);assert sha(path)==census['pools']['B-base']['sha256']
rows={}
for line in path.open('rb'):
 row=compact_row(json.loads(line));rows[row.row_id]=row
replay=D/'replay-admission-r4/candidate-replay.jsonl';assert sha(replay)=='affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a'
for line in replay.open('rb'):
 row=compact_row(json.loads(line));rows[row.row_id]=row
transformed=off_identity_occurrences(schedule,rows)
assert all(not x['enable_thinking'] for x in transformed if x['source']=='identity')
for orig,new in zip(schedule,transformed,strict=True):
 assert all(orig[k]==new[k] for k in orig if k not in ('row_id','enable_thinking'))
 if orig['source']=='replay' or not orig['enable_thinking']:assert orig==new
out=D/'ablation-a11-r1/A11';report=json.loads((out/'exposure.json').read_bytes()) if (out/'exposure.json').exists() else pack_prefix(rows,transformed,out,truncate=True,updates=72)
consumed=[json.loads(x) for x in (out/'occurrence-ledger.jsonl').read_bytes().splitlines()];assert consumed==transformed[:len(consumed)]
parentdir=D/'bfcl-ablation-r1/A3'
if not parentdir.exists():
 candidates=list(D.glob('**/A3/packed.parquet'));print('A3candidates',candidates);parentdir=candidates[0].parent
parent=pq.read_table(parentdir/'packed.parquet').to_pylist();record=pq.read_table(out/'packed.parquet').to_pylist();assert len(record)==576 and record[:280]==parent[:280]
changes=[{'sequence':i,'parent_valid':sum(x!=-1 for x in parent[i]['segment_ids']),'new_valid':sum(x!=-1 for x in record[i]['segment_ids']),'parent_supervision':sum(parent[i]['target_weights'][1:]),'new_supervision':sum(record[i]['target_weights'][1:])} for i in range(280,288) if record[i]!=parent[i]]
mass=[sum(sum(x['target_weights'][1:]) for x in record[i:i+8]) for i in range(0,576,8)];assert min(mass)>0
identity_ids={v['row_id'] for v in transformed if v['source']=='identity'}
seqs=[json.loads(x) for x in (out/'sequence-rows.jsonl').read_bytes().splitlines()];identityupdates=[i//8+1 for i,x in enumerate(seqs) if any(r in identity_ids for r in x['ids'])]
report.update(arm='A11',parent_full_r5_ledger_sha256=sha(ledger),parent_identity_pool_sha256=census['pools']['B-base']['sha256'],parent_replay_pool_sha256=sha(replay),parent_A3_packed_path=str(parentdir/'packed.parquet'),parent_A3_packed_sha256=sha(parentdir/'packed.parquet'),consumed_parent_occurrences=len(consumed),unconsumed_parent_occurrences=len(schedule)-len(consumed),first35_updates_arrays_exact_A3=True,update36_changed_sequences=changes,batch_shifted_supervision_mass=mass,each_of72_batches_nonempty=True,first_identity_update=min(identityupdates),last_identity_update=max(identityupdates),converted_ON_occurrences=sum(x['source']=='identity' and x['enable_thinking'] for x in schedule[:len(consumed)]),schedule_authority='Entire frozen r5Base10 source occurrence order; every identity ON converted same-source exact OFF; maximal72; no cycling/refill/new sources; not equal dose vsA9')
assert report['maximum_identity_source_occurrences_across_modes']<=3
(out/'exposure.json').write_text(json.dumps(report,indent=2)+'\n');(P/'data-counts.json').write_text(json.dumps({'A11':report,'A3':json.loads((parentdir/'exposure.json').read_bytes()),'A9':json.loads((D/'ablation-a9-r1/A9/exposure.json').read_bytes())},indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['row_occurrences','final_text_tokens','reasoning_text_tokens','raw_supervised_tokens','padding_only_sequences','update36_changed_sequences','converted_ON_occurrences','maximum_identity_source_occurrences_across_modes']},indent=2))
