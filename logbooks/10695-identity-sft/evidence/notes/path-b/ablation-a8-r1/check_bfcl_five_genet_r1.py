"""Independent local complete-original BFCL joins; no bootstrap, model or remote calls."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from datetime import datetime,timezone
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=['A8a','A8b'],required=True);parser.add_argument('--index',type=Path,required=True);parser.add_argument('--index-sha256',required=True);parser.add_argument('--handoff-sha256',required=True);parser.add_argument('--scores',required=True);args=parser.parse_args()
HERE=Path(__file__).resolve().parent;p=HERE/args.arm
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
assert sha(args.index)==args.index_sha256;idx=json.loads(args.index.read_text());rows=idx['runs'];assert len(rows)==len(idx['physical_runs'])==5 and idx['ALL_FIVE_COMPLETE']
plan=json.loads((HERE/'launch-plan.json').read_text());assert sha(HERE/'launch-plan.json')=='f527fb76e294211e33e5e28eb1183bc6788ae5414f229542e047c410332e2a65';target=next(a for a in plan['arms'] if a['arm']==args.arm)
agg_path=p/'all-five-BFCL-plus-original-identity132-launch-margay-r1.json';agg=json.loads(agg_path.read_text())
source=Path(idx['task_ids_only_join_source']);expected=[x['doc_id'] for x in json.loads(source.read_text())['rows']];assert len(expected)==len(set(expected))==123
base=json.loads((HERE.parent/'ablation-a6-r1/BFCL-five-r1/run-1/native-record.actual.json').read_text())
scores=[];proofs=[];config=None;digest=None
for n,x in enumerate(rows,1):
 assert x['arm']==args.arm and x['evaluation_index']==n and x['state']=='COMPLETE123' and x['replacement_number']==0 and x['replaces_incomplete_job'] is None
 assert x['full_physical_chain_retained']==[x['run_id']]
 for key in ['launch','native_record','report','array']:assert sha(x[key])==x[key+'_sha256'],(n,key)
 launch,native,report,array=[json.loads(Path(x[k]).read_text()) for k in ['launch','native_record','report','array']]
 assert x['run_id']==launch['job']==native['jobs']['orchestrator']==report['run_id']==array['run_id']==agg['BFCL'][n-1]['job']
 assert native['run_id']==report['eval_run_id']==x['eval_run_id']
 assert native['results_path']==report['results_uri']==array['results_uri']==x['results_uri']
 assert array['score_sha256']==x['report_sha256'] and report['native_record_sha256']==x['native_record_sha256']
 ids=[r['doc_id'] for r in array['rows']];assert ids==expected and len(set(ids))==len(ids)==123 and all(r['reward'] in (0,1) for r in array['rows'])
 correct=sum(r['reward'] for r in array['rows']);assert correct==x['score']==x['correct']==report['correct'] and report['n']==123
 assert math.isclose(report['mean_pct'],correct/123*100)
 assert native['status']=='succeeded' and native['error'] is None
 cov=native['coverage']['bfcl_parity'];assert (cov['n_benchmark'],cov['n_attempted'],cov['n_scored'])==(123,123,123) and not cov['errors']
 assert native['metrics']['bfcl_parity']['solved']==correct and native['metrics']['bfcl_parity']['total']==123
 assert native['model']['location']==launch['HF']==x['HF']==target['final_checkpoint']['hf_export']
 assert sha(launch['model_yaml'])==launch['model_config_sha256']==report['source_model_yaml_sha256']==x['model_config_sha256']==target['bfcl']['model_config_sha256']
 assert native['model']['config_digest']==report['native_model_config_sha256']==x['native_model_config_sha256']
 if config is None:config=native['model']['config'];digest=native['model']['config_digest']
 assert config==native['model']['config'] and digest==native['model']['config_digest']
 assert {k:v for k,v in config.items() if k not in ['name','location']}=={k:v for k,v in base['model']['config'].items() if k not in ['name','location']}
 assert native['eval']==base['eval']
 assert native['provenance']['git_sha']==launch['method_pin']==report['method_pin']==x['method_pin']=='caba0fa38b780af42da54af2f188ba5bfcc28501'
 assert launch['application_submit_calls']==1 and launch['records_prefix']==x['records_prefix']
 scores.append(int(correct));proofs.append(dict(run_id=x['run_id'],score=int(correct),array_sha256=x['array_sha256'],report_sha256=x['report_sha256'],native_record_sha256=x['native_record_sha256'],launch_sha256=x['launch_sha256']))
assert len({x['run_id'] for x in rows})==len({x['records_prefix'] for x in rows})==5
hand=p/'all-five-FULL123-operator-statistic-handoff-margay-r1.json';assert sha(hand)==args.handoff_sha256;h=json.loads(hand.read_text());assert scores==[int(s) for s in args.scores.split(',')]==h['correct_by_slot']
pcts=[s/123*100 for s in scores];assert math.isclose(statistics.mean(pcts),h['mean_pct']) and math.isclose(statistics.stdev(pcts),h['sample_SD_pp'])
cohort=Path(agg['original_slot1_diagnostic_cohort']);assert sha(cohort)==agg['original_slot1_diagnostic_cohort_sha256'];c=json.loads(cohort.read_text());assert c['actual_original_job_id']==rows[0]['run_id'] and c['doc_ids']==expected
out=p/'all-five-BFCL-independent-joins-genet-r1.json';assert not out.exists()
r=dict(utc=datetime.now(timezone.utc).isoformat(),arm=args.arm,status='PASS_ALL_FIVE_FULL123_SOURCE_LAUNCH_NATIVE_ARRAY_JOINS',index_sha256=sha(args.index),logical_runs=5,physical_runs=5,replacements=0,reward_rows=615,task_ids=123,task_ids_source_sha256=sha(source),scores=scores,mean_pct=statistics.mean(pcts),sample_sd_pp=statistics.stdev(pcts),range_pct=[min(pcts),max(pcts)],uniform_native_model_config_sha256=digest,yaml_sha256=target['bfcl']['model_config_sha256'],all5_eval_and_non_target_model_config_equal_A6=True,original_slot1_cohort_sha256=sha(cohort),joins=proofs,limits='Independent local completed-arm joins only. Frozen joint d6ba bootstrap waits both A8 arms full123x5; no partial A8a mean/CI, old34 pooling, item pairing, outcome-added A8a-A8b contrast, extra model/remote calls or release/course authority.',model_remote_calls=0,scratch_schema_note='Initial author checker expected old A6 operator-summary field names; corrected to A8 correct_by_slot/sample_SD_pp. Native source/reward/launch joins passed; no production, raw, scoring or sampling change.')
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('SHA256',sha(out))
