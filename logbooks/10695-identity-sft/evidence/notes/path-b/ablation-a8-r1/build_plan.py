"""Bind two exact A6-recipe arms with nested occurrence-level GLM ON doses."""
import copy,hashlib,json,subprocess
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
B=HERE.parent
ROOT=Path('/home/karan/oa/marin-path-b-a8-on-share')
DATA=Path('/mnt/storage/karan-scratch/genet-path-b/ablation-a8-r1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
 parent=B/'ablation-a6-r1/launch-plan.json';old=json.loads(parent.read_text());template=old['arms'][0]
 stage=json.loads((HERE/'stage-files.json').read_text());arms=[]
 for label in ['A8a','A8b']:
  target=copy.deepcopy(template);report=json.loads((DATA/label/'exposure.json').read_text())
  name=f'pathb-bfcl-{label.lower()}-r1-{pin[:12]}'
  root=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-{pin[:12]}/{label}/run'
  packed=next(x for x in stage['files'] if x['local']==str(DATA/label/'packed.parquet'))
  exposure=next(x for x in stage['files'] if x['local']==str(DATA/label/'exposure.json'))
  training=copy.deepcopy(template['training_kwargs']);training.update(run_root=root,run_id=name,arm=label,training_pin=pin,packed_path=packed['uri'],packed_sha256=packed['sha256'],exposure_path=exposure['uri'],exposure_sha256=exposure['sha256'])
  event=copy.deepcopy(template['final_checkpoint']);event.update(native_checkpoint=root+'/checkpoints/step-35',hf_export=root+'/hf/updates-36',eval_scratch=root+'/eval/updates-36',exposure=report['cumulative_batches'][-1],training_pin=pin,packed_data_sha256=packed['sha256'],arm=label)
  method=copy.deepcopy(template['bfcl']);old_yaml=Path(method['model_config']);parsed=yaml.safe_load(old_yaml.read_text());newname=f'snowball-pathb-bfcl-ablation-{label.lower()}-u36-{pin[:12]}'
  text=old_yaml.read_text().replace('name: '+parsed['name'],'name: '+newname,1).replace('location: '+parsed['location'],'location: '+event['hf_export'],1)
  dest=HERE/f'model-{label}.yaml';dest.write_text(text);new=yaml.safe_load(text)
  assert {k:v for k,v in parsed.items() if k not in ['name','location']}=={k:v for k,v in new.items() if k not in ['name','location']}
  method.update(model_config=str(dest),model_config_sha256=sha(dest),records_prefix=root+'/bfcl/records',MARIN_PREFIX=root+'/bfcl/native')
  for flag,val in {'--model-config':str(dest),'--version':newname,'--records-prefix':method['records_prefix'],'--description':f'BFCL {label} completed36 auditedHF nestedONshare'}.items():method['argv'][method['argv'].index(flag)+1]=val
  method['paired_reader_command_template']=method['paired_reader_command_template'].replace('--label pathb-bfcl-ablation-a6-u36-cc2c0160d01b','--label '+newname)
  method['paired_reader_command_template']=method['paired_reader_command_template'].replace(str(B/'ablation-a6-r1/A6/paired-score.json'),str(HERE/label/'paired-score.json'))
  method['legacy_paired_reader_is_primary']=False
  method['decision_analysis']='Five whole123 run means; frozen run-level bootstrap, freshv0.1 primary/A3+A6 secondary. Historical2130 paired reader not a decision analysis.'
  target.update(arm=label,root_name=name,training_kwargs=training,final_checkpoint=event,bfcl=method,actual_dose={k:report[k] for k in ['final_text_tokens','raw_supervised_tokens','reasoning_text_tokens','weighted_mass','structural_target_tokens']},selected_ON_occurrences=report['retained_traced_ON_occurrences'],padding_only_sequences=report['padding_only_sequences'],all36_batches_nonempty_supervision=True)
  arms.append(target)
 plan=copy.deepcopy(old);plan.update(code_pin=pin,workspace=str(ROOT),authority='Jerboa A8a/A8b ON-share GO: 179/537 nested occurrence-level subset of A6 consumed1791, exact7266ledger,36updates no old175suffix/refill; both roots together on MartencombinedCLEAR',status='READY_FOR_COMBINED_SCIENCE_READ_NO_SUBMIT_YET',arms=arms,current_input_manifest={'path':str(HERE/'stage-files.json'),'sha256':sha(HERE/'stage-files.json')},parent_A6={'path':str(parent),'sha256':sha(parent)},expected_H100h='Two36update training arms plus one-node exports,5BFCL+identity132 each; actual costs recorded separately',comparison_limits=['A8a179/A8b537 are floor10%/30% of consumedA6 traced1791 encounters,notunique-source shares','Exact7266 occurrence ledger; selectedGLMON bytes unchanged; remainingtracedON same-sourceA3OFF; allA6OFF/fallback/replay bytes/order unchanged','Canonical identityF126492/replayF616153 identicaltoA6; A3 originalprefix differs so compare actualdoses','Padding-only7/6sequences;each36logicalbatch nonempty;per-updategrouping/doses differ','No old175suffix/refill/resampling/newgeneration/target edits','Original132screen noP1prefill, BFCLfivewhole123 no scoredrun rerun/partialrepair/zeroimputation'])
 stage_receipt=HERE/'six-inputs-staging-FULLGET-readback-margay-r1.json'
 if stage_receipt.exists():plan['current_staging_receipt']={'path':str(stage_receipt),'sha256':sha(stage_receipt)}
 else:plan['current_staging_receipt']={'status':'PENDING_CPU_STAGE_ONLY_NO_SUBMIT'}
 spec=B/'bfcl-noise-r1/A8-run-comparison-preregistration-marten-r1.json';assert sha(spec)=='d6ba2832c39370a53674fd4bfb29b0dc3d9861558b945f69f0af6a11bb161f18'
 plan['BFCL'].update(primary_reference='five fresh v0.1 complete runs, unchanged priorreference data',analysis_spec={'path':str(spec),'sha256':sha(spec)},incomplete_run_rule='One wholefresh replacement per terminal incomplete original run, freshnamespace; no valid123rerun, partialrepair, zeroimputation orsecondreplacement')
 plan['identity_screen'].update(scope='exact core52+dev14 BOTH modes132; originalac986callback; noP1prefill orparser changes; target eachauditedA8HFonly')
 plan['dose_comparison']={a:json.loads((Path('/mnt/storage/karan-scratch/genet-path-b/bfcl-ablation-r1')/a/'exposure.json').read_text())['cumulative_batches'][-1] for a in ['A0','A3']}
 plan['dose_comparison']['A6']=json.loads((Path('/mnt/storage/karan-scratch/genet-path-b/ablation-a6-r1/A6/exposure.json')).read_text())['cumulative_batches'][-1]
 for arm in arms:plan['dose_comparison'][arm['arm']]=arm['final_checkpoint']['exposure']
 (HERE/'launch-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');print(pin,sha(HERE/'launch-plan.json'))
if __name__=='__main__':main()
