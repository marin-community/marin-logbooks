from pathlib import Path
import hashlib,json
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def check(plan):
 label=next(iter(plan['targets']));t=plan['targets'][label];ref=plan['qualification_aggregate'];assert sha(Path(ref['path']))==ref['sha256'];q=json.loads(Path(ref['path']).read_bytes());co=plan['original_slot1_cohort'];assert sha(Path(co['path']))==co['sha256'];cohort=json.loads(Path(co['path']).read_bytes());ids=set(cohort['all123_doc_ids']);assert len(ids)==123 and cohort['frozen_before_submit_AND_any_transcript_read']is True;results=[]
 for ref in plan['qualified_originals']:
  assert ref in q['infrastructure_incomplete_originals']and ref['already_R1_submitted']is False and ref['arm']==label;ep=Path(ref['path']);assert sha(ep)==ref['sha256'];ev=json.loads(ep.read_bytes());assert ev['SDK_terminal']is True and ev['logical_full123_admissible']is False and ev['score']is None and ev['source_HF']==t['location']and ev['arm']==label and ev['original_slot']==ref['original_slot']and ev['replacement_number']==0 and 0<=ev['scored_binary_rewards']<123;assert {x['doc_id']for x in ev['rows']}==ids and sum(x['reward']is not None for x in ev['rows'])==ev['scored_binary_rewards'];sp=ev['samples_full_JSON_native_reader_projection'];assert sha(Path(sp['path']))==sp['sha256'];samples=json.loads(Path(sp['path']).read_bytes());assert len(samples)==123 and {x['doc_id']for x in samples}==ids;expected={x['doc_id']:x['error']for x in ref['unscored_infrastructure_errors']};missing=[];scored_agent_errors=[]
  for row in samples:
   detail=json.loads(row['grading']['detail'])
   if detail['scored']is False:
    missing.append(row['doc_id']);assert detail['error']==expected[row['doc_id']]and detail['error']['category']=='infrastructure'and row['grading']['score']is None and row['metrics']==[]
   else:
    assert row['grading']['score']in [0.,1.]and detail['rewards']['reward']==row['grading']['score']
    if detail.get('error'):scored_agent_errors.append(dict(doc_id=row['doc_id'],reward=row['grading']['score'],error=detail['error']))
  assert set(missing)==set(ref['missing_NULL_task_ids'])and len(missing)==123-ev['scored_binary_rewards'];fresh=next(x for x in t['runs']if x['run']==ref['original_slot']);old=fresh['original_configuration'];assert fresh['replacement_number']==1 and fresh['run']==old['run']and fresh['version']!=old['version']and fresh['MARIN_PREFIX']!=old['MARIN_PREFIX']and fresh['records_prefix']!=old['records_prefix'];owned=ev['owned_terminal_attempts_and_scoped_cost'];assert all(x['canonical_terminal']is True for x in owned['jobs']);results.append(dict(original_slot=ref['original_slot'],source_proof_sha256=ref['sha256'],actual_native_missing_infrastructure_NULLs=missing,scored_agent_errors_preserved=scored_agent_errors))
 assert {x['original_slot']for x in results}=={x['run']for x in t['runs']};return results
