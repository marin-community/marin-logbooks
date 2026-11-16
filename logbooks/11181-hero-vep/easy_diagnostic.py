"""Small frozen answer-format diagnostic; no DNA inputs or model-specific secrets."""
import argparse
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path

QUESTIONS = [
 ('arithmetic', 'What is 2 + 3?', '5', '8'),
 ('arithmetic', 'What is 9 - 4?', '5', '6'),
 ('arithmetic', 'What is 3 multiplied by 4?', '12', '7'),
 ('arithmetic', 'Which number is larger: 7 or 2?', '7', '2'),
 ('facts', 'How many sides does a triangle have?', '3', '4'),
 ('facts', 'Which planet do humans live on?', 'Earth', 'Mars'),
 ('facts', 'How many days are in a standard week?', '7', '10'),
 ('facts', 'Which animal is a mammal?', 'Dog', 'Trout'),
 ('text', 'What is the first letter of the word cat?', 'c', 't'),
 ('text', 'What is the word dog spelled backwards?', 'god', 'dog'),
 ('text', 'How many letters are in the word moon?', '4', '3'),
 ('text', 'Which word appears in this exact text: red blue?', 'blue', 'green'),
 ('unknown', 'A fair coin was tossed privately. No outcome was provided. What was the outcome?', 'Heads', 'Tails'),
 ('unknown', 'A fair six-sided die was rolled privately. No outcome was provided. Was the result odd or even?', 'Odd', 'Even'),
 ('unknown', 'A bag contains one red and one blue ball. One was drawn privately. No outcome was provided. Which color was drawn?', 'Red', 'Blue'),
 ('unknown', 'A computer privately selected 0 or 1 with equal probability. No outcome was provided. Which value was selected?', '0', '1'),
]

def prepare(tokenizer_path, template_path):
 from tokenizers import Tokenizer
 from jinja2 import Environment, StrictUndefined
 tok = Tokenizer.from_file(str(tokenizer_path))
 text = template_path.read_text().strip()
 template = Environment(undefined=StrictUndefined).from_string(text.replace('{% generation %}', '').replace('{% endgeneration %}', ''))
 cases = []
 for qi, (kind, question, first, second) in enumerate(QUESTIONS):
  # Within each category, alternate list order, independently of content assignment.
  order = ['A','B'] if qi % 2 == 0 else ['B','A']
  for swapped in [False, True]:
   opts = dict(zip(['A','B'], [second,first] if swapped else [first,second]))
   user = 'Choose one of the two options. Reply with only the letter A or B.\n\n' + question + '\n' + '\n'.join(f'{letter}. {opts[letter]}' for letter in order)
   chat = template.render(messages=[{'role':'user','content':user}], bos_token='<|begin_of_text|>', add_generation_prompt=True)
   for arm in (['prefill'] if kind == 'unknown' else ['prefill','native']):
    prompt = chat + ('Answer:' if arm == 'prefill' else '')
    ids = tok.encode(prompt, add_special_tokens=False).ids
    candidates = [' A',' B'] if arm == 'prefill' else ['A','B']
    tids = []
    for answer in candidates:
     full = tok.encode(prompt+answer, add_special_tokens=False).ids
     assert full[:-1] == ids
     tids.append(full[-1])
    assert ids.count(128000) == 1
    assert tids == ([362,426] if arm == 'prefill' else [32,33])
    cases.append(dict(index=len(cases),question_index=qi,kind=kind,question=question,swapped=swapped,
      order=order,options=opts,correct=None if kind=='unknown' else ('B' if swapped else 'A'),arm=arm,
      prompt=prompt,prompt_ids=ids,candidates=candidates,candidate_ids=tids))
 assert len(cases)==56
 return dict(protocol='easy-diagnostic-v1',template_sha256=hashlib.sha256(text.encode()).hexdigest(),
   tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),cases=cases,max_http_requests=225,max_seconds=600)

def run(client, plan, emit):
 emit(dict(type='start',protocol=plan['protocol']))
 for case in plan['cases']:
  for suffix, expected in [('',case['prompt_ids'])]+[(s,case['prompt_ids']+[t]) for s,t in zip(case['candidates'],case['candidate_ids'])]:
   got=client.request('tokenize',dict(model=client.model,prompt=case['prompt']+suffix,add_special_tokens=False))
   assert got['tokens']==expected
  start=time.monotonic()
  response=client.request('v1/completions',dict(model=client.model,prompt=case['prompt_ids'],add_special_tokens=False,
    max_tokens=1 if case['arm']=='prefill' else 16,temperature=0,echo=False,logprobs=1,
    logprob_token_ids=case['candidate_ids'],skip_special_tokens=False,return_tokens_as_token_ids=True,
    stop_token_ids=[128001,128009],seed=11181))
  elapsed=time.monotonic()-start
  assert response['model']==client.model
  if client.fingerprint is None: client.fingerprint=response['system_fingerprint']
  assert response['system_fingerprint']==client.fingerprint
  assert response['usage']['prompt_tokens']==len(case['prompt_ids'])
  choice=response['choices'][0]
  lp=choice['logprobs']['top_logprobs'][0]
  a,b=[lp[f'token_id:{t}'] for t in case['candidate_ids']]
  assert math.isfinite(a) and math.isfinite(b)
  emit(dict(type='score',index=case['index'],logp_A=a,logp_B=b,text=choice['text'],
     finish_reason=choice['finish_reason'],completion_tokens=response['usage']['completion_tokens'],elapsed=elapsed))
  assert elapsed<=10, 'Completion latency guard'
  time.sleep(1)
 emit(dict(type='complete',responses=56))

def describe(xs):
 ys=sorted(xs)
 return dict(n=len(xs),min=min(xs),median=st.median(xs),max=max(xs),mean=st.mean(xs),sd=st.pstdev(xs))

def existing(scores, inputs):
 import gzip
 rows=[json.loads(l) for l in scores.read_text().splitlines() if json.loads(l)['type']=='score']
 assert len(rows)==400
 plan=json.loads(gzip.decompress(inputs.read_bytes()))
 out={}
 for method in ['zero-shot','two-shot']:
  rs=[r for r in rows if r['method']==method]
  margins=[r['logp_B']-r['logp_A'] for r in rs]
  paired={}
  for r in rs: paired.setdefault(r['variant'],{})[r['strand']]=r['logp_B']-r['logp_A']
  demos=[]
  if method=='two-shot':
   for r in rs:
    prompt=plan['cases'][r['index']]['prompt']
    answers=[l.split(': ')[1] for l in prompt.splitlines() if l in ['Answer: A','Answer: B']]
    assert len(answers)==2
    preferred='B' if r['logp_B']>r['logp_A'] else 'A' if r['logp_A']>r['logp_B'] else 'tie'
    demos.append(dict(last=answers[-1],preferred=preferred))
  out[method]=dict(margin_B_minus_A=describe(margins),conditional_P_B=describe([1/(1+math.exp(-x)) for x in margins]),
    strand_abs_margin_change=describe([abs(v['+']-v['-']) for v in paired.values()]),
    strand_preference_changes=sum((v['+']>0)!=(v['-']>0) for v in paired.values()),
    last_demo_breakdown={k:dict(n=sum(d['last']==k for d in demos),matches=sum(d['last']==k and d['preferred']==k for d in demos),ties=sum(d['last']==k and d['preferred']=='tie' for d in demos)) for k in ['A','B']} if demos else None)
 return out

def analyze(plan, path):
 allrows=[json.loads(l) for l in path.read_text().splitlines()]
 assert allrows[-1]['type']=='complete'
 rows=[r for r in allrows if r['type']=='score']
 assert [r['index'] for r in rows]==list(range(56))
 combined=[]
 for r,c in zip(rows,plan['cases']):
  pref='A' if r['logp_A']>r['logp_B'] else 'B' if r['logp_B']>r['logp_A'] else 'tie'
  # Strip only terminal markers/whitespace. Anything beyond a lone A/B is noncompliant.
  rendered=r['text'].replace('<|eot_id|>','').replace('<|end_of_text|>','').strip()
  combined.append(dict(**r,**{k:c[k] for k in ['question_index','kind','arm','correct','order','swapped']},
    preferred=pref,parsed_answer=rendered if rendered in ['A','B'] else None,
    conditional_P_B=1/(1+math.exp(r['logp_A']-r['logp_B'])),mass_AB=math.exp(r['logp_A'])+math.exp(r['logp_B'])))
 summary={}
 for name,rs in [('easy_prefill',[r for r in combined if r['kind']!='unknown' and r['arm']=='prefill']),
   ('easy_native',[r for r in combined if r['arm']=='native']),('unknown_prefill',[r for r in combined if r['kind']=='unknown'])]:
  summary[name]=dict(n=len(rs),candidate_correct=sum(r['preferred']==r['correct'] for r in rs),B_preferred=sum(r['preferred']=='B' for r in rs),
    last_preferred=sum(r['preferred']==r['order'][-1] for r in rs),parsed_correct=sum(r['parsed_answer']==r['correct'] for r in rs if r['correct']),
    parseable=sum(r['parsed_answer'] is not None for r in rs),conditional_P_B=describe([r['conditional_P_B'] for r in rs]),mass_AB=describe([r['mass_AB'] for r in rs]),
    by_order={''.join(o):dict(n=sum(r['order']==o for r in rs),B_preferred=sum(r['order']==o and r['preferred']=='B' for r in rs)) for o in [['A','B'],['B','A']]})
 return dict(summary=summary,cases=combined)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','existing','analyze']);p.add_argument('inputs',nargs='+',type=Path);a=p.parse_args()
 if a.mode=='prepare': result=prepare(*a.inputs[:2])
 elif a.mode=='existing': result=existing(*a.inputs[:2])
 else: result=analyze(json.loads(a.inputs[0].read_text()),a.inputs[1])
 a.inputs[-1].write_text(json.dumps(result,indent=2)+'\n')
