from pathlib import Path
import hashlib,json
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def check(plan):
 """R2: each chain's ORIGINAL and whole R1 must both be actual terminal infrastructure-incomplete (census pending entry)."""
 label=next(iter(plan['targets']));t=plan['targets'][label];assert plan['replacement_number']==2 and plan['NO_R3']is True
 ref=plan['qualification_aggregate'];assert sha(Path(ref['path']))==ref['sha256'];census=json.loads(Path(ref['path']).read_bytes())
 pend={(c['arm'],c['original_slot']):c for c in census['pending_BFCL_logical_chains']}
 co=plan['original_slot1_cohort'];assert sha(Path(co['path']))==co['sha256'];cohort=json.loads(Path(co['path']).read_bytes());ids=set(cohort['all123_doc_ids']);assert len(ids)==123 and cohort['frozen_before_submit_AND_any_transcript_read']is True
 results=[]
 for ch in plan['qualified_chains']:
  s=ch['original_slot'];assert ch['arm']==label and ch==pend[(label,s)] and ch['primary_score']is None and ch['state'].startswith('ORIGINAL_AND_R1_INFRA_INCOMPLETE')
  assert [x['replacement_number']for x in ch['physical']]==[0,1]
  fresh=next(x for x in t['runs']if x['run']==s);old=fresh['original_configuration'];r1=fresh['replacement1_configuration']
  assert fresh['replacement_number']==2 and fresh['run']==old['run']==r1['run']==s
  for k in ('version','MARIN_PREFIX','records_prefix'):assert fresh[k]!=old[k] and fresh[k]!=r1[k] and old[k]!=r1[k]
  links=[]
  for ph in ch['physical']:
   ep=Path(ph['proof']);assert sha(ep)==ph['proof_sha256'];ev=json.loads(ep.read_bytes())
   assert ev['SDK_terminal']is True and ev['logical_full123_admissible']is False and ev['score']is None and ev['source_HF']==t['location']and ev['arm']==label and ev['original_slot']==s and ev['replacement_number']==ph['replacement_number']and 0<=ev['scored_binary_rewards']<123 and ev['scored_binary_rewards']==ph['scored_binary_rewards']
   assert {x['doc_id']for x in ev['rows']}==ids and sum(x['reward']is not None for x in ev['rows'])==ev['scored_binary_rewards']
   sp=ev['samples_full_JSON_native_reader_projection'];assert sha(Path(sp['path']))==sp['sha256'];samples=json.loads(Path(sp['path']).read_bytes());assert len(samples)==123 and {x['doc_id']for x in samples}==ids
   missing=[];scored_agent_errors=[]
   for row in samples:
    detail=json.loads(row['grading']['detail'])
    if detail['scored']is False:
     missing.append(row['doc_id']);assert detail['error']['category']=='infrastructure'and row['grading']['score']is None and row['metrics']==[]
    else:
     assert row['grading']['score']in [0.,1.]and detail['rewards']['reward']==row['grading']['score']
     if detail.get('error'):scored_agent_errors.append(dict(doc_id=row['doc_id'],reward=row['grading']['score'],error=detail['error']))
   assert len(missing)==123-ev['scored_binary_rewards']>=1
   owned=ev['owned_terminal_attempts_and_scoped_cost'];assert owned['jobs']and all(x['canonical_terminal']is True for x in owned['jobs'])  # SDK succeeded is NOT required (M49 s1 R1 root worker_failed)
   links.append(dict(replacement_number=ph['replacement_number'],root_job=ph['root_job'],proof=ph['proof'],proof_sha256=ph['proof_sha256'],scored_binary_rewards=ev['scored_binary_rewards'],actual_native_missing_infrastructure_NULLs=missing,scored_agent_errors_preserved=scored_agent_errors))
  results.append(dict(original_slot=s,chain=links))
 assert {x['original_slot']for x in results}=={x['run']for x in t['runs']};return results
