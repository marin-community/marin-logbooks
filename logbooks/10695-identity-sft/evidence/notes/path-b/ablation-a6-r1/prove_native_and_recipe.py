"""All288 native examples and exact numerical recipe comparisons, CPU only."""
from pathlib import Path
from dataclasses import replace,asdict
import hashlib,json
import draccus,haliax as hax,numpy as np,pyarrow.parquet as pq
from levanter.data.text.datasets import LmDataConfig
from experiments.post_training.identity_path_b.ablation import AblationArm,config_for,check_report
from experiments.post_training.identity_path_b.recipe import path_b_config
from experiments.post_training.identity_path_b.native import packed_dataset
from experiments.post_training.identity_t3.train import t3_config
from experiments.post_training.identity_context_distill.runtime_recipe import configure_precision_runtime
HERE=Path(__file__).resolve().parent
OUT=Path('/mnt/storage/karan-scratch/genet-path-b/ablation-a6-r1')
N=HERE.parent.parent/'t310'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 results=[]
 for arm in [AblationArm.GLM_ON_OFF_FALLBACK]:
  folder=OUT/arm;raw=(folder/'packed.parquet').read_bytes();records=pq.read_table(folder/'packed.parquet').to_pylist()
  report=json.loads((folder/'exposure.json').read_text());check_report(report,arm)
  dataset=packed_dataset(raw,hashlib.sha256(raw).hexdigest(),hax.Axis('position',8192)).as_sync_dataset()
  mass=0
  for offset in range(0,288,8):
   examples=dataset.get_batch(list(range(offset,offset+8)))
   for index,example in enumerate(examples,offset):
    row=records[index]
    np.testing.assert_array_equal(example.tokens.array,row['input_ids'])
    expected=np.roll(np.asarray(row['target_weights'],dtype=np.float32),-1);expected[-1]=0
    expected*=np.roll(np.asarray(row['segment_ids']),-1)>=0
    np.testing.assert_array_equal(example.loss_weight.array,expected)
    query,keys=example.attn_mask.segment_ids
    np.testing.assert_array_equal(query.array,row['segment_ids']);np.testing.assert_array_equal(keys.array,row['segment_ids'])
    mass+=float(expected.sum())
  assert mass==sum(report['raw_supervised_tokens'].values())
  results.append({'arm':arm,'native_examples':len(records),'actual_shifted_weight_sum':mass,'all_tokens_weights_query_key':'PASS'})
 source=(N/'receipt-inputs/config.json').read_bytes();generation=(N/'receipt-inputs/generation_config.json').read_bytes();rounding=(N/'receipt-inputs/routing.json').read_bytes()
 receipt=json.loads((N/'runtime-recipe-02036367-torch-order-xla-default.json').read_text())
 assert hashlib.sha256(source).hexdigest()==receipt['source_metadata']['config_sha256']
 assert hashlib.sha256(generation).hexdigest()==receipt['source_metadata']['generation_config_sha256']
 assert hashlib.sha256(rounding).hexdigest()==receipt['routing_reference']['sha256']
 uri='s3://marin-rno2a/tmp/ttl=14d/bfcl-ablation-fixture/recipe'
 base=path_b_config(data=LmDataConfig(tokenizer='passthrough',vocab_size=128256,components={}),updates=36,output_uri=uri,run_id='fixture',source_config=json.loads(source),source_generation_config=json.loads(generation))
 t3=t3_config(uri,'fixture',source,generation,(N/'cd16-t3/freeze-preview-r2/data-config.json').read_bytes(),receipt)
 common=dict(config_bytes=source,generation_bytes=generation,rounding_bytes=rounding,rounding_path=receipt['routing_reference']['path'],rounding_sha256=receipt['routing_reference']['sha256'],xla_precision='default')
 configured,observed=configure_precision_runtime(base,**common);old,oldobserved=configure_precision_runtime(t3,**common)
 assert observed==oldobserved==receipt
 for arm in [AblationArm.GLM_ON_OFF_FALLBACK]:
  candidate=config_for(configured,arm)
  normalization=replace(candidate,data=old.data,hf_save_steps=old.hf_save_steps,adapter=old.adapter,
    trainer=replace(candidate.trainer,num_train_steps=old.trainer.num_train_steps,checkpointer=old.trainer.checkpointer))
  assert normalization==old
  if arm==AblationArm.T3_BRIDGE:assert candidate.adapter==old.adapter
  a=[float(candidate.optimizer.lr_scheduler(36)(i)) for i in range(36)]
  b=[float(old.optimizer.lr_scheduler(16)(i)) for i in range(36)]
  c=[float(base.optimizer.lr_scheduler(140)(i)) for i in range(36)]
  np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)
 (HERE/'all-native-and-recipe-proof.json').write_text(json.dumps({'native':results,'A6_same_numeric_recipe_as_A0':True,'normalized_non_science_fields':['explicit36horizon','directfinalonlysave/hfcallbackpolicy','actualdatareader frozen arrays'],'all36_lr_vs140_and16_equal':a,'live_calls':0},indent=2)+'\n')
 print(json.dumps(results))
if __name__=='__main__':main()
