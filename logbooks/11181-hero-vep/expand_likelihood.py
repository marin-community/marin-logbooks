"""Expand existing train pilots to 30 matched groups; guarded reuse and checkpointed scoring."""
import argparse,collections,hashlib,json,math,time
from pathlib import Path
from prompts import HEALTHY_PREFIX

def prepare(inputs,old_missense,old_tss,missense_scores,tss_scores,output):
 from prepare import INPUTS,AUTOSOMES,read_rows,rank,variant_id,attach_sequences
 for name,h in INPUTS.items():
  with (inputs/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
 oldm=json.loads(old_missense.read_text());oldt=json.loads(old_tss.read_text())
 old={'missense_variant':oldm['panels']['pilot'],'tss_proximal':oldt['panels']['pilot']}
 groups={s:collections.defaultdict(list) for s in old}
 cols=['chrom','pos','ref','alt','label','subset','match_group','consequence','distance_tss']
 for r in read_rows(inputs/'variants-train.parquet',cols):
  if r['chrom'] in AUTOSOMES and r['subset'] in groups:groups[r['subset']][str(r['match_group'])].append(r)
 panels={};summary={}
 for subset in old:
  prior=old[subset];oldgroups=list(dict.fromkeys(str(r['match_group']) for r in prior));assert len(oldgroups)==10
  for g,rs in groups[subset].items():
   assert len(rs)==10 and sum(r['label'] for r in rs)==1 and len({r['chrom'] for r in rs})==1
   rs.sort(key=variant_id)
  ns='query-group' if subset=='missense_variant' else 'tss-proximal-group'
  newgroups=sorted((g for g in groups[subset] if g not in oldgroups),key=lambda g:rank(11181,ns,g))[:20]
  new=[r for g in newgroups for r in groups[subset][g]];attach_sequences({'added':new},inputs/'sequences-train.parquet')
  panels[subset]=prior+new
  assert len({r['id'] for r in panels[subset]})==300
  priorvalidation={r['id'] for r in oldm['panels'].get('validation',[])}
  summary[subset]=dict(old_groups=oldgroups,added_groups=newgroups,eligible_groups=len(groups[subset]),added_variants=200,
   added_from_previously_reserved_missense_panel=sum(r['id'] in priorvalidation for r in new),chromosomes=sorted({r['chrom'] for r in panels[subset]},key=int))
 scoremaps={}
 for subset,path in [('missense_variant',missense_scores),('tss_proximal',tss_scores)]:
  rs=[json.loads(l) for l in path.read_text().splitlines()]
  assert rs[-1]['type']=='complete'
  scoremaps[subset]={(r['variant'],r['method'],r['strand'],r.get('allele',r.get('candidate'))):r for r in rs if r['type']=='score' and r['method'] in ['null','healthy']}
  assert len(scoremaps[subset])==800
 tasks=[];checks=[]
 for subset,rows in panels.items():
  for i,row in enumerate(rows):
   for strand in ['+','-']:
    for method,prefix in [('null',''),('healthy',HEALTHY_PREFIX)]:
     for allele in ['reference','alternate']:
      text=row['sequences'][strand][allele];h=hashlib.sha256((prefix+text).encode()).hexdigest()
      t=dict(index=len(tasks),subset=subset,variant=row['id'],strand=strand,method=method,allele=allele,prefix=prefix,text=text,input_sha256=h,existing=i<100)
      if i<100:
       r=scoremaps[subset][row['id'],method,strand,allele];assert math.isfinite(r['logprob'])
       if 'input_sha256' in r:assert r['input_sha256']==h
       t['historical_logprob']=r['logprob']
      tasks.append(t)
      if i in [0,99]:checks.append(t['index'])
 assert len(tasks)==4800 and len(checks)==32
 data=dict(protocol='expanded300-likelihood-v1',seed=11181,split='train',panels=panels,selection=summary,tasks=tasks,compatibility_indices=checks,
  max_total_seconds=9000,max_total_http=14000,chunk_targets=400,methods=['null','healthy'],source_hashes={str(path.name):hashlib.sha256(path.read_bytes()).hexdigest() for path in [old_missense,old_tss,missense_scores,tss_scores]})
 output.write_text(json.dumps(data,indent=2)+'\n')

def score(client,t):
 result=client.score(t['prefix'],t['text'],False)
 return dict(**{k:t[k] for k in ['index','subset','variant','strand','method','allele','input_sha256']},**result)

def compatibility(client,plan,emit):
 vals={};reuse={s:True for s in plan['panels']}
 for i in plan['compatibility_indices']:
  t=plan['tasks'][i];r=score(client,t);r.update(type='compatibility',historical_delta=r['logprob']-t['historical_logprob']);emit(r);vals[i]=r['logprob']
  reuse[t['subset']] &= abs(r['historical_delta'])<=1e-6
 for i in [plan['compatibility_indices'][j] for j in [0,2,4,6,16,18,20,22]]:
  r=score(client,plan['tasks'][i]);r.update(type='repeat',delta=abs(r['logprob']-vals[i]));emit(r);assert r['delta']<=1e-6
 emit(dict(type='compatibility_complete',reuse=reuse))

def chunk(client,plan,indices,anchors,emit):
 emit(dict(type='start',indices=indices))
 def check():
  for i,expected in anchors.items():
   r=score(client,plan['tasks'][int(i)]);r.update(type='repeat',delta=abs(r['logprob']-expected));emit(r);assert r['delta']<=1e-6
 check()
 for n,i in enumerate(indices,1):
  r=score(client,plan['tasks'][i]);r['type']='score';emit(r)
  if n%80==0:check()
 if len(indices)%80:check()
 emit(dict(type='complete',targets=len(indices)))

def aggregate(plan,compat,chunk_paths):
 cr=[json.loads(l) for l in compat.read_text().splitlines()];assert cr[-1]['type']=='compatibility_complete'
 reuse=cr[-1]['reuse'];lookup={}
 for t in plan['tasks']:
  if t['existing'] and reuse[t['subset']]:lookup[t['index']]=t['historical_logprob']
 for path in chunk_paths:
  rs=[json.loads(l) for l in path.read_text().splitlines()];assert rs[-1]['type']=='complete'
  assert rs[-1]['targets']==len([r for r in rs if r['type']=='score'])
  for r in rs:
   if r['type']=='repeat':assert r['delta']<=1e-6
   if r['type']=='score':
    t=plan['tasks'][r['index']];assert r['index'] not in lookup and math.isfinite(r['logprob'])
    assert all(r[k]==t[k] for k in ['subset','variant','strand','method','allele','input_sha256'])
    lookup[r['index']]=r['logprob']
 assert set(lookup)==set(range(4800))
 keyed={(t['subset'],t['variant'],t['method'],t['strand'],t['allele']):lookup[t['index']] for t in plan['tasks']}
 out=[]
 for subset,rows in plan['panels'].items():
  for row in rows:
   r={k:row[k] for k in ['id','subset','match_group','label','chrom','pos']}
   for m in ['null','healthy']:r[m]=-sum(keyed[subset,row['id'],m,s,'alternate']-keyed[subset,row['id'],m,s,'reference'] for s in ['+','-'])/2
   out.append(r)
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('paths',nargs='+',type=Path);a=p.parse_args()
 if a.mode=='prepare':prepare(*a.paths)
 else:
  data=aggregate(json.loads(a.paths[0].read_text()),a.paths[1],a.paths[2:-1]);a.paths[-1].write_text(json.dumps(data,indent=2)+'\n')
