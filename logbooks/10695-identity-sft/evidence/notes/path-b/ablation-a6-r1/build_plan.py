"""Freeze five final-only training/export requests plus unchanged BFCL bindings."""
from pathlib import Path
import copy,hashlib,json,subprocess
import yaml
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/karan/oa/marin-path-b-a6-glm-off')
N=HERE.parent.parent
DATA=Path('/mnt/storage/karan-scratch/genet-path-b/ablation-a6-r1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 from experiments.post_training.identity_path_b.launch import SOURCE_RECEIPT,SOURCE_OBJECTS,RUNTIME_SHA256
 pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
 parent=HERE.parent/'r5-weight1-r1/launch-plan.json';old=json.loads(parent.read_text())
 inputs=json.loads((HERE/'stage-files.json').read_text())
 source=HERE.parent/'full-screen-execution-r1/execution-packet.json'
 assert sha(source)=='13b058213e399e402d36a3f1c5612dc885b4345381939e36aaec351268deebb2'
 template=next(a for a in json.loads(source.read_text())['candidates'] if a['label']=='pathb-base25-u52-r1')
 bfcl=template['bfcl'];assert sha(Path(bfcl['model_config']))==bfcl['model_config_sha256']
 assert sha(Path(bfcl['policy_wrapper']))==bfcl['policy_wrapper_sha256']
 arms=[]
 for arm in ['A6']:
  folder=DATA/arm;report=json.loads((folder/'exposure.json').read_text())
  name=f'pathb-bfcl-{arm.lower()}-r1-{pin[:12]}'
  root=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-{pin[:12]}/{arm}/run'
  packed=next(f for f in inputs['files'] if f['local']==str(folder/'packed.parquet'))
  exposure=next(f for f in inputs['files'] if f['local']==str(folder/'exposure.json'))
  training=dict(run_root=root,run_id=name,arm=arm,training_pin=pin,packed_path=packed['uri'],packed_sha256=packed['sha256'],exposure_path=exposure['uri'],exposure_sha256=exposure['sha256'],runtime_sha256=RUNTIME_SHA256,source_receipt=SOURCE_RECEIPT,source_objects=SOURCE_OBJECTS)
  final=dict(completed_updates=36,native_checkpoint=root+'/checkpoints/step-35',hf_export=root+'/hf/updates-36',eval_scratch=root+'/eval/updates-36',exposure=report['cumulative_batches'][-1],training_pin=pin,packed_data_sha256=packed['sha256'],arm=arm)
  label=f'pathb-bfcl-ablation-{arm.lower()}-u36-{pin[:12]}'
  dest=HERE/f'model-{arm}.yaml'
  text=Path(bfcl['model_config']).read_text().replace('name: snowball-pathb-base25-u52-r1','name: snowball-'+label,1).replace('location: '+template['model_path'],'location: '+final['hf_export'],1)
  old_yaml=yaml.safe_load(Path(bfcl['model_config']).read_text());new_yaml=yaml.safe_load(text)
  assert {k:v for k,v in old_yaml.items() if k not in ['name','location']}=={k:v for k,v in new_yaml.items() if k not in ['name','location']}
  dest.write_text(text)
  method=copy.deepcopy(bfcl);method.update(model_config=str(dest),model_config_sha256=sha(dest),records_prefix=root+'/bfcl/records',MARIN_PREFIX=root+'/bfcl/native')
  argv=method['argv'];changes={'--model-config':str(dest),'--version':label,'--records-prefix':method['records_prefix'],'--description':f'BFCL ablation {arm} completed36 audited HF'}
  for key,value in changes.items():argv[argv.index(key)+1]=value
  method.pop('command',None)
  method['paired_reader_command_template']=bfcl['paired_reader_command_template'].replace('--label pathb-base25-u52-r1','--label '+label).replace('/mnt/storage/karan-scratch/pathb-full-screens-r1/pathb-base25-u52-r1/bfcl-paired.json',str(HERE/arm/'paired-score.json'))
  frozen=26 if arm=='A4' else 104
  arms.append(dict(arm=arm,root_name=name,training_kwargs=training,final_checkpoint=final,bfcl=method,
    actual_dose={key:report[key] for key in ['final_text_tokens','raw_supervised_tokens','reasoning_text_tokens','weighted_mass','structural_target_tokens']},
    frozen_native_tensors=frozen,trained_master_cast_tensors=502-frozen,special_rows_frozen=arm not in ['A1','A4'],
    checkpoint_updates=[36],optional_GSM=False,restore_from_checkpoint=False))
 plan=dict(schema_version=1,code_pin=pin,workspace=str(ROOT),status='READY_FOR_COMBINED_SCIENCE_READ_NO_SUBMIT_YET',authority='Jerboa A6 fixed36 maximal transformed A0 prefix; existing GLM ON else same-source OFF; no37/no refill/no wait for references after MartenCLEAR',
   runtime=old['runtime'],source_receipt=SOURCE_RECEIPT,source_objects=SOURCE_OBJECTS,
   parent_r5=dict(path=str(parent),sha256=sha(parent)),current_input_manifest=dict(path=str(HERE/'stage-files.json'),sha256=sha(HERE/'stage-files.json')),
   cpu_root=dict(cpu=8,memory='32g',disk='50g',lifetime_seconds=176400,priority='interactive',failure_retries=0,preemption_retries=0),
   training=dict(hosts=4,gpus_per_host=8,cpu_per_host=64,memory_per_host='768GB',disk_per_host='1TB',lifetime_seconds=43200,priority='interactive',failure_retries=0,preemption_retries=0,jax_platforms='cuda,cpu',batch=8,microbatch=8,sequence_length=8192,updates=36,seed=310,learning_rate=2e-6,warmup=2,schedule='constant',weight_decay=0.0,max_grad_norm=0.5,loss='sum shifted supervised CE / sum shifted binary masks across logical global batch'),
   export=dict(hosts=1,gpus=8,cpu=64,memory='768GB',disk='1TB',lifetime_seconds=7200,CPU_write_probe_before_GPU=True,native_final_only=True,temporary=True,delete_native_only_after_exact_audited_export=True),
   BFCL=dict(workspace=bfcl['workspace'],pin=bfcl['workspace_pin'],source_science_packet_sha256=sha(source),historical_reference_only='2130 34/123, not pooled with current references',primary_reference='five fresh v0.1 complete BFCL runs; comparison waits all admitted complete references',evaluation_runs=5,all_items=123,paired_bootstrap='PythonRNG297/10000',outer_CPU_lifetime_seconds=176400,native_serve_lifetime_seconds=43200,HF_only=True,native_checkpoint_leases_or_reads=0,one_completion_notifier_per_root=True),
   arms=arms,comparison_limits=['A6 finite transformed prefix needs37updates to consume whole A0 list; authorized36 maximal prefix omits175 suffix occurrences','Canonical finals remain same per occurrence; realized canonical dose differs at cutoff','Existing GLM ON targets unchanged; failed trace uses exact same-source OFF, not empty ON','No rebalance/refill or new generation','Identity132 BOTH modes follows original ac986 callback/66contexts with only audited A6 model path changed'],expected_H100h='One36-update arm plus export/BFCL5/identity132; actual costs recorded separately',remote_calls_by_genet=0)
 plan['current_staging_receipt']=dict(path=str(HERE/'three-inputs-staging-FULLGET-readback-margay-r1.json'),sha256=sha(HERE/'three-inputs-staging-FULLGET-readback-margay-r1.json'))
 plan['identity_screen']=dict(source_pin='ac986b08026c07ab5096535361bccc36bf01d553',source_packet=str(HERE.parent/'ablation-identity132-r1'),scope='exact core52+dev14 BOTH modes,132; no injected identity system; only audited A6 model path/namespace change; no math/unseen',raw_preserved=True,actual_outputs_read_by='marten')
 plan['dose_comparison']={arm:json.loads((Path('/mnt/storage/karan-scratch/genet-path-b/bfcl-ablation-r1')/arm/'exposure.json').read_text())['cumulative_batches'][-1] for arm in ['A0','A3']}
 plan['dose_comparison']['A6']=arms[0]['final_checkpoint']['exposure']
 (HERE/'launch-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');print(pin,sha(HERE/'launch-plan.json'))
if __name__=='__main__':main()
