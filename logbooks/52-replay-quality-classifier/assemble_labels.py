"""Revalidate immutable responses, measure blind agreement, and select adjudications."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from label import CATEGORIES, PROMPT


def parsed_label(row):
    if 'response' not in row:
        return None
    choice=row['response']['choices'][0]
    if choice['finish_reason']!='stop':
        return None
    try:
        label=json.loads(choice['message']['content'])
    except (TypeError,json.JSONDecodeError):
        return None
    if label.get('category') not in CATEGORIES or type(label.get('low_value')) not in (bool,type(None)):
        return None
    if (label['category']=='uncertain')!=(label['low_value'] is None):
        return None
    if not 0<=float(label['confidence'])<=1:
        return None
    return {key:label[key] for key in ('category','low_value','confidence','rationale')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root
    glm={}
    receipts={}
    statuses=Counter()
    for path in sorted((root/'labels-v1').glob('sample-*.json')):
        if '.started.' in path.name: continue
        row=json.loads(path.read_text()); statuses[row['status']]+=1
        label=parsed_label(row)
        if label is not None:
            glm[row['sample_id']]=label
            receipts[row['sample_id']]=hashlib.sha256(path.read_bytes()).hexdigest()
    astra={}
    for path in sorted((root/'astra-v1').glob('batch-*/labels.json')):
        for label in json.loads(path.read_text())['labels']:
            sample_id=label['sample_id']
            assert sample_id not in astra
            assert label['category'] in CATEGORIES
            astra[sample_id]={key:label[key] for key in ('category','low_value','confidence','rationale')}
    panel=json.loads((root/'astra-v1/panel.json').read_text())['ids']
    assert set(astra).issubset(panel)
    paired=sorted(set(glm)&set(astra))
    conclusive=[index for index in paired if glm[index]['low_value'] is not None and astra[index]['low_value'] is not None]
    agreement=sum(glm[i]['low_value']==astra[i]['low_value'] for i in conclusive)/len(conclusive) if conclusive else None
    p1=sum(glm[i]['low_value'] for i in conclusive)/len(conclusive) if conclusive else 0
    p2=sum(astra[i]['low_value'] for i in conclusive)/len(conclusive) if conclusive else 0
    chance=p1*p2+(1-p1)*(1-p2)
    summary={'glm_parsed':len(glm),'glm_status':dict(statuses),'astra_labels':len(astra),'panel_planned':len(panel),'paired':len(paired),'both_conclusive':len(conclusive),'binary_agreement':agreement,'cohen_kappa':(agreement-chance)/(1-chance) if conclusive and chance<1 else None,
             'category_agreement':sum(glm[i]['category']==astra[i]['category'] for i in paired)/len(paired) if paired else None}
    disagreements=[i for i in paired if glm[i]['category']!=astra[i]['category'] or glm[i]['low_value']!=astra[i]['low_value']]
    wrapper_cases=[i for labels in (glm,astra) for i,label in labels.items() if label['low_value'] and any(word in label['rationale'] for word in ('begin_of_text','end_of_text','special tokens','tokenizer artifact'))]
    adjudicate=sorted(set(disagreements+wrapper_cases))
    # Labels are private until aggregation/adjudication; never include text in a public report.
    (root/'label-panel-private.json').write_text(json.dumps({'glm':glm,'astra':astra,'receipts':receipts,'adjudicate':adjudicate,'summary':summary},indent=2))
    print(json.dumps({**summary,'category_or_binary_disagreements':len(disagreements),'wrapper_audit_cases':len(set(wrapper_cases)),'adjudication_cases':len(adjudicate)},indent=2))

if __name__=='__main__': main()
