"""Open the frozen test only after committing model selection and calibration."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from assemble_labels import parsed_label
from train import cutoff, metrics


def interval(hits, total):
    z=1.95996398454
    p=hits/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [float(center-radius),float(center+radius)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    args=parser.parse_args(); root=args.root; output=args.model_dir
    manifest=json.loads((root/'sample-v1/manifest.json').read_text())
    labels={}
    for row in manifest['samples']:
        path=root/'labels-v1'/f"sample-{row['sample_id']:04d}.json"
        if not path.exists(): continue
        label=parsed_label(json.loads(path.read_text()))
        if label is not None: labels[row['sample_id']]=label
    for path in sorted((root/'adjudication-v1').glob('batch-*/labels.json')):
        for label in json.loads(path.read_text())['labels']: labels[label['sample_id']]=label
    samples={row['sample_id']:row for row in manifest['samples']}
    test=sorted(i for i,label in labels.items() if samples[i]['split']=='test' and label['low_value'] is not None)
    calibration=sorted(i for i,label in labels.items() if samples[i]['split']=='calibration' and label['low_value'] is not None)
    model_calibration=json.loads((output/'calibration.json').read_text())
    assert calibration==model_calibration['calibration_ids']
    assert not set(test)&set(model_calibration['training_ids']+calibration)
    frozen_model_sha=hashlib.sha256((output/'model.json').read_bytes()).hexdigest()
    assert frozen_model_sha==model_calibration['model_sha256']
    all_scores=np.load(output/'scores.npz')
    structural=np.load(root/'structural-v1.npz')
    junk=json.loads((root/'junkgate-scores-v1.json').read_text())
    scores={'learned':dict(zip(map(int,all_scores['ids']),map(float,all_scores['score']),strict=True)),
            'rules':dict(zip(map(int,structural['ids']),map(float,structural['rules']),strict=True)),
            'junkgate2':{row['sample_id']:-row['score'] for row in junk['scores']}}
    y=np.array([labels[i]['low_value'] for i in test],dtype=bool)
    cy=np.array([labels[i]['low_value'] for i in calibration],dtype=bool)
    reports={}
    predictions={}
    for name,by_id in scores.items():
        score=np.array([by_id[i] for i in test])
        calibration_score=np.array([by_id[i] for i in calibration])
        thresholds={}
        for limit in (.02,.05):
            threshold=model_calibration['thresholds'][str(limit)]['cutoff'] if name=='learned' else cutoff(calibration_score,cy,limit)
            result=metrics(score,y,threshold)
            result['recall_interval95']=interval(result['true_drops'],result['low_value'])
            result['false_drop_interval95']=interval(result['false_drops'],result['natural'])
            result['cutoff']=threshold
            result['calibration']=metrics(calibration_score,cy,threshold)
            thresholds[str(limit)]=result
            predictions[(name,limit)]=score>=threshold
        precision,recall,pr_thresholds=precision_recall_curve(y,score)
        reports[name]={'average_precision':float(average_precision_score(y,score)),'operating_points':thresholds,
                       'pr_curve':{'precision':precision.tolist(),'recall':recall.tolist(),'thresholds':pr_thresholds.tolist()}}
    rng=np.random.default_rng(52611200)
    improvements={}
    for baseline in ('junkgate2','rules'):
        for limit in (.02,.05):
            difference=(predictions[('learned',limit)].astype(int)-predictions[(baseline,limit)].astype(int))[y]
            # Resample the same documents for both classifiers; thresholds remain frozen.
            boot=np.mean(difference[rng.integers(len(difference),size=(10000,len(difference)))],axis=1)
            improvements[f'{baseline}@{limit}']={'recall_difference':float(difference.mean()),'paired_interval95':np.quantile(boot,[.025,.975]).tolist()}
    panel=json.loads((root/'label-panel-private.json').read_text())
    summary={'model_sha256':frozen_model_sha,'calibration_sha256':hashlib.sha256((output/'calibration.json').read_bytes()).hexdigest(),
             'manifest_sha256':hashlib.sha256((root/'sample-v1/manifest.json').read_bytes()).hexdigest(),
             'test_ids':test,'test_planned':sum(row['split']=='test' for row in samples.values()),
             'label_counts':dict(Counter(label['category'] for label in labels.values())),
             'independent_agreement_before_adjudication':panel['summary'],
             'adjudicated':sum(len(json.loads(path.read_text())['labels']) for path in (root/'adjudication-v1').glob('batch-*/labels.json')),
             'models':reports,'paired_recall_improvement':improvements,
             'limitations':['Model-assisted labels, not human ground truth.','Bounded views can miss content between recorded windows.','Rates are unweighted on a balanced, capacity-capped cache sample; they do not establish corpus prevalence.','Grouping uses exact visible-content hashes; near duplicates can cross splits.','The same independent model also adjudicates in fresh contexts.','Astra used the initial rubric; GLM continuation explicitly excludes universal tokenizer wrappers as evidence.']}
    (output/'evaluation.json').write_text(json.dumps(summary,indent=2))
    public=[]
    for i in sorted(samples):
        label=labels.get(i)
        public.append({'sample_id':i,'text_sha256':samples[i]['text_sha256'],'split':samples[i]['split'],
                       'category':label['category'] if label else 'unavailable','low_value':label['low_value'] if label else None})
    (output/'label-manifest.json').write_text(json.dumps(public,indent=2))
    print(json.dumps({'test_documents':len(test),'test_low_value':int(y.sum()),'test_natural':int((~y).sum()),
                      'results':{name:row['operating_points'] for name,row in reports.items()},'paired_improvements':improvements},indent=2))

if __name__=='__main__': main()
