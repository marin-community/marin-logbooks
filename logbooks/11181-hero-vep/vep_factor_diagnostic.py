"""Fully crossed four-variant prompt sensitivity diagnostic; no benchmark accuracy claims."""
import argparse, collections, hashlib, json, math, random, re, statistics, time
from pathlib import Path


def prepare(source_path, tokenizer_path, template_path):
 from tokenizers import Tokenizer
 from jinja2 import Environment, StrictUndefined
 source=json.loads(source_path.read_text())
 base={}
 for c in source['cases']:
  if c['method']=='zero-shot' and c['arm']=='raw' and c['mapping']=='primary' and not c['repeat']:
   base.setdefault(c['variant'],c)
 assert len(base)==4 and all(v.split(':')[0] not in ['X','chrX','23'] for v in base)
 tok=Tokenizer.from_file(str(tokenizer_path))
 text=template_path.read_text().strip()
 template=Environment(undefined=StrictUndefined).from_string(text.replace('{% generation %}','').replace('{% endgeneration %}',''))
 cases=[]
 for vi,(variant,c) in enumerate(base.items()):
  raw=c['prompt'];seq=re.search(r'>sequence\n([ACGT]+)\n',raw).group(1)
  assert len(seq)==255 and raw.endswith('\nAnswer:')
  rng=random.Random(11181+vi)
  left=list(seq[:127]);right=list(seq[128:]);rng.shuffle(left);rng.shuffle(right)
  shuffled=''.join(left)+seq[127]+''.join(right)
  assert shuffled!=seq and shuffled[127]==seq[127] and collections.Counter(seq)==collections.Counter(shuffled)
  for symbols in ['AB','CD']:
   for mapping in ['primary','reversed']:
    meanings=['pathogenic','benign'] if mapping=='primary' else ['benign','pathogenic']
    options=dict(zip(symbols,meanings))
    for order in ['forward','reverse']:
     displayed=list(symbols if order=='forward' else symbols[::-1])
     for dna,dna_seq in [('real',seq),('shuffled',shuffled)]:
      user=raw.removesuffix('\nAnswer:').replace('A: pathogenic\nB: benign','\n'.join(f'{s}: {options[s]}' for s in displayed))
      user=user.replace('Respond with exactly one letter: A or B.',f'Respond with exactly one letter: {symbols[0]} or {symbols[1]}.')
      assert user.count(seq)==1
      user=user.replace(seq,dna_seq)
      prompt=template.render(messages=[{'role':'user','content':user}],bos_token='<|begin_of_text|>',add_generation_prompt=True)+'Answer:'
      ids=tok.encode(prompt,add_special_tokens=False).ids
      assert ids.count(128000)==1
      tids=[]
      for s in symbols:
       full=tok.encode(prompt+' '+s,add_special_tokens=False).ids
       assert full[:-1]==ids
       tids.append(full[-1])
      assert len(set(tids))==2
      cases.append(dict(variant=variant,source_label=c['label'],match_group=c['match_group'],symbols=symbols,mapping=mapping,
       order=order,displayed=displayed,dna=dna,options=options,prompt=prompt,prompt_ids=ids,candidate_ids=tids))
 assert len(cases)==64
 random.Random(11181).shuffle(cases)
 for i,c in enumerate(cases): c['index']=i
 repeats=[next(c['index'] for c in cases if c['variant']==v and c['symbols']=='AB' and c['mapping']=='primary' and c['order']=='forward' and c['dna']=='real') for v in base]
 return dict(protocol='vep-factor-v1',cases=cases,repeat_indices=repeats,seed=11181,max_seconds=600,max_http_requests=261,
  source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),template_sha256=hashlib.sha256(text.encode()).hexdigest())


def run(client,plan,emit):
 baselines={}
 emit(dict(type='start',protocol=plan['protocol']))
 for c,repeat in [(c,False) for c in plan['cases']]+[(plan['cases'][i],True) for i in plan['repeat_indices']]:
  if not repeat:
   for suffix,expected in [('',c['prompt_ids'])]+[(' '+s,c['prompt_ids']+[t]) for s,t in zip(c['symbols'],c['candidate_ids'])]:
    got=client.request('tokenize',dict(model=client.model,prompt=c['prompt']+suffix,add_special_tokens=False))
    assert got['tokens']==expected
  start=time.monotonic()
  response=client.request('v1/completions',dict(model=client.model,prompt=c['prompt_ids'],add_special_tokens=False,max_tokens=1,
   temperature=1.0,seed=11181,echo=False,logprobs=1,logprob_token_ids=c['candidate_ids'],ignore_eos=True,
   skip_special_tokens=False,return_tokens_as_token_ids=True))
  elapsed=time.monotonic()-start
  assert response['model']==client.model and response['usage']['prompt_tokens']==len(c['prompt_ids'])
  assert response['usage']['completion_tokens']==1
  if client.fingerprint is None:client.fingerprint=response['system_fingerprint']
  assert client.fingerprint==response['system_fingerprint']
  choice=response['choices'][0];lp=choice['logprobs']['top_logprobs'][0]
  values=[lp[f'token_id:{t}'] for t in c['candidate_ids']]
  assert all(math.isfinite(x) for x in values)
  row=dict(type='repeat' if repeat else 'score',index=c['index'],logprobs=values,elapsed=elapsed,sampled_text=choice['text'])
  if repeat:row['max_delta']=max(abs(x-y) for x,y in zip(values,baselines[c['index']]))
  emit(row)
  if repeat:assert row['max_delta']<=1e-5
  else:baselines[c['index']]=values
  assert elapsed<=10,'Completion latency guard'
  time.sleep(1)
 emit(dict(type='complete',scores=64,repeats=4))


def stats(xs):
 return dict(n=len(xs),min=min(xs),median=statistics.median(xs),max=max(xs),mean=statistics.mean(xs))


def analyze(plan,path):
 rows=[json.loads(l) for l in path.read_text().splitlines()]
 assert rows[-1]['type']=='complete'
 scores=[r for r in rows if r['type']=='score'];repeats=[r for r in rows if r['type']=='repeat']
 assert [r['index'] for r in scores]==list(range(64))
 assert len(repeats)==4 and max(r['max_delta'] for r in repeats)<=1e-5
 combined=[]
 for c,r in zip(plan['cases'],scores):
  margin=r['logprobs'][1]-r['logprobs'][0]
  pref=c['symbols'][1] if margin>0 else c['symbols'][0] if margin<0 else 'tie'
  pm=margin if c['mapping']=='reversed' else -margin
  combined.append(dict(**{k:c[k] for k in ['index','variant','symbols','mapping','order','displayed','dna','source_label']},
   margin_second_minus_first=margin,pathogenic_margin=pm,conditional_P_pathogenic=1/(1+math.exp(-pm)),
   preferred=pref,semantic_preference='tie' if pref=='tie' else c['options'][pref],
   last_preferred=pref==c['displayed'][-1],mass_candidates=sum(math.exp(x) for x in r['logprobs'])))
 slices={}
 for symbols in ['AB','CD']:
  for dna in ['real','shuffled']:
   rs=[r for r in combined if r['symbols']==symbols and r['dna']==dna]
   slices[symbols+'_'+dna]=dict(n=len(rs),second_symbol_preferred=sum(r['preferred']==symbols[1] for r in rs),
    last_preferred=sum(r['last_preferred'] for r in rs),pathogenic_preferred=sum(r['semantic_preference']=='pathogenic' for r in rs),ties=sum(r['preferred']=='tie' for r in rs),
    mean_candidate_mass=statistics.mean(r['mass_candidates'] for r in rs))
 effects={}
 dims=['variant','symbols','mapping','order','dna']
 for dim,levels in [('order',['forward','reverse']),('symbols',['AB','CD']),('dna',['real','shuffled']),('mapping',['primary','reversed'])]:
  keys=[x for x in dims if x!=dim];groups={}
  for r in combined:groups.setdefault(tuple(r[k] for k in keys),{})[r[dim]]=r
  pairs=[]
  for group in groups.values():
   a,b=[group[x] for x in levels]
   pairs.append(dict(context={k:a[k] for k in keys},delta_pathogenic_margin=b['pathogenic_margin']-a['pathogenic_margin'],
    delta_conditional_P_pathogenic=b['conditional_P_pathogenic']-a['conditional_P_pathogenic'],
    semantic_preference_changed=a['semantic_preference']!=b['semantic_preference']))
  assert len(pairs)==32
  effects[dim]=dict(direction=f'{levels[1]} minus {levels[0]}',n=32,semantic_changes=sum(p['semantic_preference_changed'] for p in pairs),
   abs_margin_delta=stats([abs(p['delta_pathogenic_margin']) for p in pairs]),signed_margin_delta=stats([p['delta_pathogenic_margin'] for p in pairs]),
   abs_probability_delta=stats([abs(p['delta_conditional_P_pathogenic']) for p in pairs]),pairs=pairs)
 return dict(slices=slices,effects=effects,cases=combined,repeat_max_delta=max(r['max_delta'] for r in repeats))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','analyze']);p.add_argument('paths',nargs='+',type=Path);a=p.parse_args()
 result=prepare(*a.paths[:3]) if a.mode=='prepare' else analyze(json.loads(a.paths[0].read_text()),a.paths[1])
 a.paths[-1].write_text(json.dumps(result,indent=2)+'\n')
