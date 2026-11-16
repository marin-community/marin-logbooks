# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Validate complete chat pilot evidence and compute pinned group metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import marin_dna_evals.metrics as native

METRICS_SHA256 = 'bee720324d61ea8b88a4644b6bd036569e9bab6f8e931d6335d165fb2e2b2395'


def report(plan_path, panels_path, scores_path, historical_path):
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == METRICS_SHA256
    plan=json.loads(plan_path.read_text());panels=json.loads(panels_path.read_text())
    assert plan['panel_sha256']==hashlib.sha256(panels_path.read_bytes()).hexdigest()
    records=[json.loads(l) for l in scores_path.open()]
    assert records[0]['type']=='start' and records[-1]['type']=='complete'
    assert records[0]['panel_sha256']==plan['panel_sha256'] and records[0]['protocol']=='chat-prefill-v1'
    scores={};guards=[]
    for r in records[1:-1]:
        c=plan['cases'][r['index']]
        for k in ['variant','method','strand','mapping']:assert r[k]==c[k]
        assert r['prompt_sha256']==hashlib.sha256(c['prompt'].encode()).hexdigest()
        assert all(math.isfinite(r[k]) for k in ['logp_A','logp_B'])
        if r['type']=='repeat':
            baseline=scores[r['index']]
            delta=max(abs(r[k]-baseline[k]) for k in ['logp_A','logp_B'])
            assert delta==r['max_logprob_delta'] and delta<=1e-5
            guards.append(r)
        else:
            assert r['type']=='score' and r['index'] not in scores
            scores[r['index']]=r
    assert set(scores)==set(range(400)) and len(guards)==8
    expected_repeats=plan['repeat_indices']*len(plan['repeat_after'])
    assert [r['index'] for r in guards]==expected_repeats
    bykey={(r['variant'],r['method'],r['strand']):r for r in scores.values()}
    historical={r['id']:r for r in json.loads(historical_path.read_text())}
    variants=[]
    for row in panels['panels']['pilot']:
        v={k:row[k] for k in ['id','label','subset','match_group','chrom','answer_mapping']}
        old=historical[row['id']]
        for k in v:assert old[k]==v[k]
        for method in ['zero-shot','two-shot']:
            margins=[]
            for strand in ['+','-']:
                r=bykey[(row['id'],method,strand)]
                margins.append((r['logp_A']-r['logp_B'])*(1 if row['answer_mapping']=='primary' else -1))
            v[method]=sum(margins)/2
            v[method+'/strand-difference']=margins[0]-margins[1]
            v['historical-raw/'+method]=old[method]
        variants.append(v)
    f=pd.DataFrame(variants); grouped=f.groupby('match_group')
    assert len(f)==100 and grouped.ngroups==10
    assert (grouped.size()==10).all() and (grouped.label.sum()==1).all()
    result={'n_variants':100,'n_groups':10,'exploratory':True,'bootstrap_replicates':2000,
            'bootstrap_seed':11181,'metrics_source_sha256':METRICS_SHA256,
            'repeat_max_logprob_delta':max(r['max_logprob_delta'] for r in guards),'methods':{},'paired_deltas':{}}
    for method in ['zero-shot','two-shot']:
        rr=[r for r in scores.values() if r['method']==method]
        mass=[math.exp(r['logp_A'])+math.exp(r['logp_B']) for r in rr]
        wins=[]
        for _,g in grouped:
            pos=g.loc[g.label.astype(bool),method].iloc[0]
            controls=g.loc[~g.label.astype(bool),method]
            wins.extend((pos>controls).astype(float)+.5*(pos==controls))
        m={'auprc':native.auprc_with_bootstrap_se(f.label,f[method],f.match_group,n_bootstrap=2000,rng=11181),
           'historical_raw_auprc':float(average_precision_score(f.label,f['historical-raw/'+method])),
           'within_group_pairwise_accuracy':float(np.mean(wins)),
           'answer_mass':{'mean':float(np.mean(mass)),'min':min(mass),'max':max(mass)},
           'n_contexts':len(rr),'A_preferred':sum(r['logp_A']>r['logp_B'] for r in rr),
           'B_preferred':sum(r['logp_B']>r['logp_A'] for r in rr),
           'ties':sum(r['logp_A']==r['logp_B'] for r in rr),'mapping_slices':{}}
        for mapping,g in f.groupby('answer_mapping'):
            rs=[r for r in rr if r['mapping']==mapping]
            m['mapping_slices'][mapping]={'n_variants':len(g),'n_positive':int(g.label.sum()),
                'prevalence':float(g.label.mean()),'auprc_descriptive':float(average_precision_score(g.label,g[method])),
                'n_contexts':len(rs),'A_preferred':sum(r['logp_A']>r['logp_B'] for r in rs),
                'B_preferred':sum(r['logp_B']>r['logp_A'] for r in rs)}
        result['methods'][method]=m
        result['paired_deltas']['chat minus historical raw / '+method]=native.paired_metric_delta_bootstrap(
            f.label,f[method],f['historical-raw/'+method],f.match_group,n_bootstrap=2000,rng=11181)
    result['paired_deltas']['two-shot minus zero-shot']=native.paired_metric_delta_bootstrap(
        f.label,f['two-shot'],f['zero-shot'],f.match_group,n_bootstrap=2000,rng=11181)
    return variants,result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ['plan','panels','scores','historical','variants_output','metrics_output']:p.add_argument('--'+arg.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    variants,metrics=report(a.plan,a.panels,a.scores,a.historical)
    for path,value in [(a.variants_output,variants),(a.metrics_output,metrics)]:
        with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(metrics,indent=2))
