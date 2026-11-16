"""Authored review controls and independently varied instruction label order."""
import argparse,json,math,random,statistics,hashlib
from pathlib import Path
from vep_factor_diagnostic import run as factor_run

REVIEWS=[('positive','It works perfectly. I love it and would happily buy it again.'),('positive','Excellent quality, fast delivery, and very helpful support. Highly recommended.'),('negative','It broke immediately, and support refused to help. I regret buying it.'),('negative','Terrible quality and a complete waste of money. I would never recommend it.'),('ambiguous','The item arrived on Tuesday. I have not opened or used it yet.'),('ambiguous','Some features are good and some are bad. Overall I have mixed feelings about it.')]

def prepare(root,tokpath):
 from tokenizers import Tokenizer
 from jinja2 import Environment,StrictUndefined
 tok=Tokenizer.from_file(str(tokpath));tp=(root/'templates/marin-chat-pinned.jinja').read_text().strip()
 template=Environment(undefined=StrictUndefined).from_string(tp.replace('{% generation %}','').replace('{% endgeneration %}',''))
 cases=[]
 def add(c,prompt):
  ids=tok.encode(prompt,add_special_tokens=False).ids
  assert ids.count(128000)==1
  tids=[]
  for answer in [' A',' B']:
   full=tok.encode(prompt+answer,add_special_tokens=False).ids;assert full[:-1]==ids;tids.append(full[-1])
  assert tids==[362,426]
  cases.append(dict(**c,prompt=prompt,prompt_ids=ids,candidate_ids=tids,symbols='AB'))
 for i,(label,review) in enumerate(REVIEWS):
  for mapping in ['primary','reversed']:
   opts=dict(zip('AB',['positive','negative'] if mapping=='primary' else ['negative','positive']))
   for order in ['forward','reverse']:
    for mention in ['AB','BA']:
     for context in ['short','structured']:
      metadata='' if context=='short' else 'Unrelated record metadata (not evidence about sentiment):\n```text\n'+ '\n'.join(f'record_id=R{j:03d}; batch_code=ZX{j:02d}; field_code=Q{j:02d}; row_index={j}' for j in range(1,13))+'\n```\n\n'
      user='Classify the customer review as positive or negative.\n'+'\n'.join(f'{s}: {opts[s]}' for s in ('AB' if order=='forward' else 'BA'))+f'\nRespond with exactly one letter: {mention[0]} or {mention[1]}.\n\n'+metadata+'Review: '+review+'\nWhich sentiment applies to this review?'
      prompt=template.render(messages=[{'role':'user','content':user}],bos_token='<|begin_of_text|>',add_generation_prompt=True)+'Answer:'
      add(dict(domain='review',item=str(i),label=label,target='positive',mapping=mapping,order=order,mention=mention,context=context),prompt)
 old=json.loads((root/'vep-factor-inputs.json').read_text())
 for c in old['cases']:
  if c['symbols']=='AB' and c['dna']=='real':
   for mention in ['AB','BA']:
    prompt=c['prompt'].replace('Respond with exactly one letter: A or B.',f'Respond with exactly one letter: {mention[0]} or {mention[1]}.')
    add(dict(domain='vep',item=c['variant'],label='not_scored',target='pathogenic',mapping=c['mapping'],order=c['order'],mention=mention,context='original'),prompt)
 assert len(cases)==128
 random.Random(11181).shuffle(cases)
 for i,c in enumerate(cases):c['index']=i
 repeats=[next(c['index'] for c in cases if c['domain']==d and c['mapping']==m and c['order']=='forward' and c['mention']=='AB' and c['context'] in ['short','original']) for d in ['review','vep'] for m in ['primary','reversed']]
 return dict(protocol='cross-domain-v1',cases=cases,repeat_indices=repeats,max_http_requests=517,max_seconds=600,
  tokenizer_sha256=hashlib.sha256(tokpath.read_bytes()).hexdigest(),template_sha256=hashlib.sha256(tp.encode()).hexdigest())

def run(client,plan,emit):
 def wrapped(row):
  if row['type']=='complete':row=dict(type='complete',scores=len(plan['cases']),repeats=len(plan['repeat_indices']))
  emit(row)
 factor_run(client,plan,wrapped)

def analyze(plan,path):
 rows=[json.loads(l) for l in path.read_text().splitlines()];assert rows[-1]==dict(type='complete',scores=128,repeats=4)
 rs=[r for r in rows if r['type']=='score'];assert [r['index'] for r in rs]==list(range(128))
 repeats=[r for r in rows if r['type']=='repeat'];assert len(repeats)==4 and max(r['max_delta'] for r in repeats)<=1e-5
 out=[]
 for c,r in zip(plan['cases'],rs):
  a,b=r['logprobs'];margin=(a-b) if c['mapping']=='primary' else (b-a)
  letter='A' if a>b else 'B' if b>a else 'tie'
  target_pref='tie' if margin==0 else c['target'] if margin>0 else ('negative' if c['domain']=='review' else 'benign')
  out.append(dict(**{k:c[k] for k in ['index','domain','item','label','target','mapping','order','mention','context']},letter=letter,target_margin=margin,p_target=1/(1+math.exp(-margin)),semantic=target_pref,mass_AB=math.exp(a)+math.exp(b)))
 slices={}
 for domain in ['review','vep']:
  for context in sorted({c['context'] for c in out if c['domain']==domain}):
   for label in sorted({c['label'] for c in out if c['domain']==domain}):
    sub=[c for c in out if (c['domain'],c['context'],c['label'])==(domain,context,label)]
    slices[f'{domain}_{context}_{label}']=dict(n=len(sub),B=sum(c['letter']=='B' for c in sub),last_mentioned=sum(c['letter']==c['mention'][-1] for c in sub),correct=None if label in ['ambiguous','not_scored'] else sum(c['semantic']==label for c in sub),mean_mass_AB=statistics.mean(c['mass_AB'] for c in sub))
 effects={}
 for domain in ['review','vep']:
  for factor in ['mapping','order','mention']+(['context'] if domain=='review' else []):
   keys=[k for k in ['item','mapping','order','mention','context'] if k!=factor];groups={}
   for r in out:
    if r['domain']==domain:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
   pairs=[]
   levels={'mapping':['primary','reversed'],'order':['forward','reverse'],'mention':['AB','BA'],'context':['short','structured']}[factor]
   for group in groups.values():
    assert len(group)==2
    a,b=[next(r for r in group if r[factor]==l) for l in levels]
    pairs.append(dict(item=a['item'],label=a['label'],context=a['context'],first_index=a['index'],second_index=b['index'],semantic_flip=a['semantic']!=b['semantic'],delta_p_target=b['p_target']-a['p_target']))
   effects[domain+'_'+factor]=dict(n=len(pairs),flips=sum(p['semantic_flip'] for p in pairs),median_abs_delta=statistics.median(abs(p['delta_p_target']) for p in pairs),pairs=pairs)
 return dict(slices=slices,effects=effects,cases=out,repeat_max_delta=max(r['max_delta'] for r in repeats))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('paths',nargs='+',type=Path);a=p.parse_args()
 result=prepare(*a.paths[:2]) if a.mode=='prepare' else analyze(json.loads(a.paths[0].read_text()),a.paths[1])
 a.paths[-1].write_text(json.dumps(result,indent=2)+'\n')
