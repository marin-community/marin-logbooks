"""Select a small head using training-only validation, then calibrate without test access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES
from assemble_labels import parsed_label


def cutoff(scores, labels, limit):
    # Higher scores are junk. Include the reject-none candidate and all score ties.
    candidates=np.r_[np.nextafter(scores.max(),np.inf),np.unique(scores)]
    allowed=[value for value in candidates if np.mean(scores[~labels]>=value)<=limit]
    return float(min(allowed))


def metrics(scores, labels, threshold):
    drop=scores>=threshold
    natural=int((~labels).sum()); low=int(labels.sum())
    tp=int((drop&labels).sum()); fp=int((drop&~labels).sum())
    return {'documents':len(labels),'low_value':low,'natural':natural,'true_drops':tp,'false_drops':fp,
            'recall':tp/low if low else None,'false_drop_rate':fp/natural if natural else None,
            'precision':tp/(tp+fp) if tp+fp else None}


def export(model, dimensions, scaler=None):
    if isinstance(model,LogisticRegression):
        return {'kind':'logistic','feature_names':list(FEATURE_NAMES),'embedding_dimensions':dimensions,
                'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficient':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])}
    probability=model.init_.predict_proba(np.zeros((1,model.n_features_in_)))[0,1]
    trees=[]
    for estimator in model.estimators_[:,0]:
        tree=estimator.tree_
        trees.append({'left':tree.children_left.tolist(),'right':tree.children_right.tolist(),'feature':tree.feature.tolist(),
                      'threshold':tree.threshold.tolist(),'value':tree.value[:,0,0].tolist()})
    return {'kind':'gradient_boosting','feature_names':list(FEATURE_NAMES),'embedding_dimensions':dimensions,
            'intercept':float(np.log(probability/(1-probability))),'learning_rate':model.learning_rate,'trees':trees}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--embeddings',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.root
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'sample-v1/manifest.json').read_text())
    # Deliberately do not load test labels here.
    samples={row['sample_id']:row for row in manifest['samples'] if row['split']!='test'}
    labels={}
    for index in samples:
        path=root/'labels-v1'/f'sample-{index:04d}.json'
        if not path.exists(): continue
        label=parsed_label(json.loads(path.read_text()))
        if label is not None and label['low_value'] is not None: labels[index]=label
    for path in sorted((root/'adjudication-v1').glob('batch-*/labels.json')):
        for label in json.loads(path.read_text())['labels']:
            index=label['sample_id']
            if index not in samples: continue
            labels.pop(index,None)
            if label['low_value'] is not None: labels[index]=label
    data=np.load(root/'structural-v1.npz')
    positions={int(index):position for position,index in enumerate(data['ids'])}
    training=sorted(index for index in labels if samples[index]['split']=='train')
    calibration=sorted(index for index in labels if samples[index]['split']=='calibration')
    inner_train=[i for i in training if int(samples[i]['text_sha256'][8:16],16)%5!=0]
    inner_val=[i for i in training if i not in inner_train]
    variants={'structural':data['features']}
    if args.embeddings:
        embed=np.load(args.embeddings)
        assert np.array_equal(data['ids'],embed['ids'])
        variants['fusion']=np.concatenate((data['features'],embed['embedding']),axis=1)
        variants['embedding']=embed['embedding']
    def x(ids,values): return values[[positions[i] for i in ids]]
    def y(ids): return np.array([labels[i]['low_value'] for i in ids],dtype=bool)
    candidates=[]
    for variant,values in variants.items():
        for depth in (2,3):
            for leaf in (10,25):
                model=GradientBoostingClassifier(n_estimators=160,max_depth=depth,min_samples_leaf=leaf,learning_rate=.05,random_state=52611)
                model.fit(x(inner_train,values),y(inner_train))
                score=model.predict_proba(x(inner_val,values))[:,1]
                recalls=[metrics(score,y(inner_val),cutoff(score,y(inner_val),limit))['recall'] for limit in (.02,.05)]
                candidates.append({'variant':variant,'kind':'gbdt','depth':depth,'leaf':leaf,'selection':float(np.mean(recalls)),
                                   'ap':float(average_precision_score(y(inner_val),score))})
        for c in (.1,1,10):
            scaler=StandardScaler().fit(x(inner_train,values))
            model=LogisticRegression(C=c,max_iter=2000,random_state=52611).fit(scaler.transform(x(inner_train,values)),y(inner_train))
            score=model.predict_proba(scaler.transform(x(inner_val,values)))[:,1]
            recalls=[metrics(score,y(inner_val),cutoff(score,y(inner_val),limit))['recall'] for limit in (.02,.05)]
            candidates.append({'variant':variant,'kind':'logistic','c':c,'selection':float(np.mean(recalls)),
                               'ap':float(average_precision_score(y(inner_val),score))})
    best=max(candidates,key=lambda row:(row['selection'],row['ap']))
    structural=max((row for row in candidates if row['variant']=='structural'),key=lambda row:(row['selection'],row['ap']))
    # Fixed cost-aware selection rule, before test evaluation.
    chosen=structural if structural['selection']>=best['selection']-.02 else best
    if chosen['variant']=='embedding':
        raise ValueError('Embedding-only export needs a separate feature schema; compare it but do not silently use the structural schema')
    values=variants[chosen['variant']]
    scaler=None
    if chosen['kind']=='gbdt':
        model=GradientBoostingClassifier(n_estimators=160,max_depth=chosen['depth'],min_samples_leaf=chosen['leaf'],learning_rate=.05,random_state=52611)
        model.fit(x(training,values),y(training))
        all_scores=model.predict_proba(values)[:,1]
    else:
        scaler=StandardScaler().fit(x(training,values))
        model=LogisticRegression(C=chosen['c'],max_iter=2000,random_state=52611).fit(scaler.transform(x(training,values)),y(training))
        all_scores=model.predict_proba(scaler.transform(values))[:,1]
    exported=export(model,values.shape[1]-len(FEATURE_NAMES),scaler)
    model_path=args.output/'model.json'; model_path.write_text(json.dumps(exported,separators=(',',':')))
    np.savez_compressed(args.output/'scores.npz',ids=data['ids'],score=all_scores)
    score=all_scores[[positions[i] for i in calibration]]
    thresholds={str(limit):{'cutoff':cutoff(score,y(calibration),limit)} for limit in (.02,.05)}
    for setting in thresholds.values(): setting['calibration']=metrics(score,y(calibration),setting['cutoff'])
    calibration_receipt={'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'comparison':'low_value_score >= cutoff',
                         'thresholds':thresholds,'training_ids':training,'calibration_ids':calibration,
                         'manifest_sha256':hashlib.sha256((root/'sample-v1/manifest.json').read_bytes()).hexdigest()}
    (args.output/'calibration.json').write_text(json.dumps(calibration_receipt,indent=2))
    (args.output/'selection.json').write_text(json.dumps({'chosen':chosen,'candidates':candidates,'inner_train':inner_train,'inner_validation':inner_val,
                                                         'rule':'prefer structural when its mean recall at 2%/5% inner-validation FPR is within .02 of best'},indent=2))
    print(json.dumps({'chosen':chosen,'training':len(training),'calibration':len(calibration),'thresholds':thresholds},indent=2))

if __name__=='__main__': main()
