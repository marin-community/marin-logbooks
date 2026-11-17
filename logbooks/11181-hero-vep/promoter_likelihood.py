"""TSS-proximal 100-variant DNA-likelihood pilot, using existing scoring semantics."""
import argparse,collections,hashlib,json,math,time
from pathlib import Path
from prompts import HEALTHY_PREFIX


def prepare(inputs,output):
 from prepare import INPUTS,AUTOSOMES,read_rows,rank,variant_id,attach_sequences
 for name,h in INPUTS.items():
  with (inputs/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
 groups=collections.defaultdict(list)
 cols=['chrom','pos','ref','alt','label','subset','match_group','consequence','consequence_final','distance_tss','distance_tss_pc','distance_tss_nc','tss_closest_gene_id']
 for row in read_rows(inputs/'variants-train.parquet',cols):
  if row['chrom'] in AUTOSOMES and row['subset']=='tss_proximal':
   assert row['consequence_final']=='tss_proximal' and 0<=row['distance_tss']<=1000
   assert row['consequence'] in ['intergenic_variant','intron_variant','upstream_gene_variant','downstream_gene_variant']
   groups[str(row['match_group'])].append(row)
 for rows in groups.values():
  assert len(rows)==10 and sum(r['label'] for r in rows)==1
  assert len({r['chrom'] for r in rows})==1
  rows.sort(key=variant_id)
 keys=sorted(groups,key=lambda g:rank(11181,'tss-proximal-group',g))[:10]
 assert len(keys)==10
 panels={'pilot':[r for g in keys for r in groups[g]]}
 attach_sequences(panels,inputs/'sequences-train.parquet')
 payload=dict(protocol='tss-proximal-likelihood-v1',seed=11181,subset='tss_proximal',genome='GRCh38',split='train',source_revision='4aed58e50c5dea0b878a665007af2ef9e5108e9f',sequence_revision='7b92f047f9a36f90e9ac47886afa2a99264ee35c',inputs_sha256=INPUTS,eligible_autosomal_groups=len(groups),selected_groups=keys,panels=panels,
   methods={'null':'','healthy':HEALTHY_PREFIX},max_http_requests=3000,max_seconds=2700,completion_gap_seconds=1,repeat_every=80)
 output.write_text(json.dumps(payload,indent=2)+'\n')


def targets(plan):
 out=[]
 for row in plan['panels']['pilot']:
  for strand in ['+','-']:
   for method,prefix in plan['methods'].items():
    for allele in ['reference','alternate']:
     out.append(dict(index=len(out),variant=row['id'],strand=strand,method=method,allele=allele,prefix=prefix,text=row['sequences'][strand][allele]))
 assert len(out)==800
 return out


def run(client,plan,emit):
 ts=targets(plan);base={}
 def score(t,repeat):
  result=client.score(t['prefix'],t['text'],False)
  r=dict(type='repeat' if repeat else 'score',**{k:t[k] for k in ['index','variant','strand','method','allele']},**result)
  assert math.isfinite(r['logprob'])
  if repeat:r['delta']=abs(r['logprob']-base[t['index']])
  emit(r)
  if repeat:assert r['delta']<=1e-6,'Repeated DNA score drift'
  elif t['index']<8:base[t['index']]=r['logprob']
 emit(dict(type='start',protocol=plan['protocol']))
 for i,t in enumerate(ts,1):
  score(t,False)
  if i%plan['repeat_every']==0:
   for anchor in ts[:8]:score(anchor,True)
 emit(dict(type='complete',scores=800,repeats=80))


def aggregate(plan,path):
 records=[json.loads(l) for l in path.read_text().splitlines()]
 assert records[-1]==dict(type='complete',scores=800,repeats=80)
 scores=[r for r in records if r['type']=='score'];assert [r['index'] for r in scores]==list(range(800))
 repeats=[r for r in records if r['type']=='repeat'];assert len(repeats)==80 and max(r['delta'] for r in repeats)<=1e-6
 ts=targets(plan)
 for t,r in zip(ts,scores):
  assert all(r[k]==t[k] for k in ['variant','strand','method','allele'])
  assert math.isfinite(r['logprob'])
 lookup={(r['variant'],r['method'],r['strand'],r['allele']):r['logprob'] for r in scores}
 out=[]
 for row in plan['panels']['pilot']:
  r={k:row[k] for k in ['id','chrom','pos','label','subset','match_group','distance_tss','consequence']}
  for method in ['null','healthy']:
   r[method]=-sum(lookup[row['id'],method,s,'alternate']-lookup[row['id'],method,s,'reference'] for s in ['+','-'])/2
  out.append(r)
 return out


def metrics(variants):
 import marin_dna_evals.metrics as native
 import numpy as np
 import pandas as pd
 assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()=='bee720324d61ea8b88a4644b6bd036569e9bab6f8e931d6335d165fb2e2b2395'
 f=pd.DataFrame(variants);g=f.groupby('match_group')
 assert len(f)==100 and f.id.is_unique and f.match_group.nunique()==10 and (g.size()==10).all() and (g.label.sum()==1).all()
 assert np.isfinite(f[['null','healthy']].to_numpy()).all()
 result=dict(n_variants=100,n_groups=10,positives=10,exploratory=True,bootstrap_replicates=2000,bootstrap_seed=11181,auprc={},within_group_pairwise_accuracy={})
 for method in ['null','healthy']:
  result['auprc'][method]=native.auprc_with_bootstrap_se(f.label,f[method],f.match_group,n_bootstrap=2000,rng=11181)
  wins=[]
  for _,grp in g:
   pos=grp.loc[grp.label,method].iloc[0];neg=grp.loc[~grp.label,method];wins.extend((pos>neg).astype(float)+0.5*(pos==neg))
  result['within_group_pairwise_accuracy'][method]=float(np.mean(wins))
 result['healthy_minus_null']=native.paired_metric_delta_bootstrap(f.label,f.healthy,f['null'],f.match_group,n_bootstrap=2000,rng=11181)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','aggregate','metrics']);p.add_argument('paths',nargs='+',type=Path);a=p.parse_args()
 if a.mode=='prepare':prepare(*a.paths)
 else:
  data=json.loads(a.paths[0].read_text());r=aggregate(data,a.paths[1]) if a.mode=='aggregate' else metrics(data)
  a.paths[-1].write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
