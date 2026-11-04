"""Continue the unchanged local T3 native cache/sampler to288 positions."""
from dataclasses import replace
from pathlib import Path
from collections import Counter
import hashlib,json
import draccus,haliax as hax,jax,numpy as np
import pyarrow as pa,pyarrow.parquet as pq
from levanter.data.text.datasets import LmDataConfig
from levanter.schedule import BatchSchedule
from levanter.tokenizers import load_tokenizer
from experiments.post_training.identity_t2.prepare import CapturedChatFormat,array_sha,summarize
from experiments.post_training.identity_path_b.exposure import target_counts
from experiments.post_training.identity_path_b.ablation_data import SCHEMA,UPDATES,SEQUENCES
N=Path('/home/karan/Dropbox/marin-notes/snowball-identity/t310')
P=N/'cd16-t3/freeze-preview-r2'
OUT=Path('/mnt/storage/karan-scratch/genet-path-b/bfcl-ablation-r1/A4')
HERE=Path(__file__).resolve().parent
TOKENIZER=Path('/mnt/storage/karan-scratch/margay-t2-client-tmp/levanter_tokenizers/s3:/marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a/hf-hub-1.21.0')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 OUT.mkdir(exist_ok=False)
 cfg=json.loads((P/'data-config.json').read_text())
 data=draccus.decode(LmDataConfig,cfg)
 # Preserve exact config identity; local tokenizer is byte-bound metadata only.
 tok=load_tokenizer(str(TOKENIZER));tok=replace(tok,_name_or_path=data.tokenizer)
 data.__dict__['the_tokenizer']=tok
 native=json.loads((P/'native-row-fingerprints.json').read_text())
 fingerprints={(r['tokens_sha256'],r['target_mask_sha256']):i for i,r in native.items()}
 dataset=data.train_set(hax.Axis('position',8192),BatchSchedule(8),key=jax.random.PRNGKey(310)).as_sync_dataset()
 sequences=[];ledger=[];raw=Counter();final=Counter();reasoning=Counter();structures=Counter();family=Counter();cumulative=[]
 for position in range(SEQUENCES):
  example=dataset.get_batch([position])[0]
  tokens=np.asarray(example.tokens.array,dtype=np.int32);weights=np.asarray(example.loss_weight.array,dtype=np.float32)
  identifier=fingerprints[(array_sha(tokens),array_sha(weights))];meta=native[identifier]
  query,keys=example.attn_mask.segment_ids;q=np.asarray(query.array);k=np.asarray(keys.array)
  np.testing.assert_array_equal(q,k)
  masks=np.zeros_like(weights);masks[1:]=weights[:-1]
  np.testing.assert_array_equal(np.concatenate([masks[1:],[0]]),weights)
  sequences.append({'input_ids':tokens.tolist(),'target_weights':masks.tolist(),'segment_ids':q.tolist()})
  dose=target_counts({'input_ids':tokens.tolist(),'assistant_masks':masks.tolist()})
  source='identity' if meta['component'] in {'identity','identity_v2'} else 'replay'
  raw[source]+=dose.raw;final[source]+=dose.final_text;reasoning[source]+=dose.reasoning_text
  structures.update({f'{source}/{token}':n for token,n in dose.structural.items()})
  family[f"{source}/{meta['family']}"]+=dose.final_text
  ledger.append({'position':position,'update':position//8,**meta})
  if (position+1)%8==0:
   cumulative.append({'completed_updates':(position+1)//8,'identity_final_text_tokens':final['identity'],'identity_supervised_tokens':raw['identity'],'all_supervised_tokens':raw.total(),'raw_supervised_tokens':dict(raw),'final_text_tokens':dict(final),'reasoning_text_tokens':dict(reasoning),'structural_target_tokens':dict(structures)})
 expected=[json.loads(l) for l in (P/'occurrences-128.jsonl').read_bytes().splitlines()]
 assert ledger[:128]==expected
 assert summarize(ledger[:128])==json.loads((N/'cd16-t3/regional-freeze-r1/freeze.json').read_text())['first128']
 pq.write_table(pa.Table.from_pylist(sequences,schema=SCHEMA),OUT/'packed.parquet',compression='zstd')
 (OUT/'occurrence-ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger))
 report={'experiment':'PathB-BFCL-ablation','arm':'A4','updates':36,'checkpoint_updates':[36],'batch_size':8,'sequence_length':8192,'special_token_loss_weight':1.0,'raw_supervised_tokens':dict(raw),'weighted_mass':dict(raw),'final_text_tokens':dict(final),'reasoning_text_tokens':dict(reasoning),'structural_target_tokens':dict(structures),'family_final_text_tokens':dict(family),'cumulative_batches':cumulative,'historical_first128_summary':summarize(ledger[:128]),'all288_summary':summarize(ledger),'data_config_sha256':sha(P/'data-config.json'),'cache_catalog_sha256':sha(P/'cache-catalog.json'),'native_fingerprints_sha256':sha(P/'native-row-fingerprints.json'),'source_tokenizer_files':{p.name:sha(p) for p in TOKENIZER.iterdir() if p.is_file()},'ordered_occurrence_ledger_sha256':sha(OUT/'occurrence-ledger.jsonl'),'first16_exact_reproduction':True,'mode':'historical_T3_unpacked_native_cache_sampler_bridge_not_one_factor_A0'}
 (OUT/'exposure.json').write_text(json.dumps(report,indent=2)+'\n');(HERE/'data-counts-A4.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['cumulative_batches','source_tokenizer_files']},indent=2))
if __name__=='__main__':main()
