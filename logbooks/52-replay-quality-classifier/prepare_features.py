"""Measure frozen windows without consulting labels or test outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from features import FEATURE_NAMES, document_features, rules_score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[json.loads(path.read_text()) for path in sorted(args.sample.glob('sample-*.json'))]
    values=[]
    rules=[]
    for row in rows:
        values.append(document_features(row['windows'],row['tokens']))
        rules.append(rules_score(row['windows']))
        if len(values)%100==0: print(json.dumps({'features':len(values)}),flush=True)
    np.savez_compressed(args.output,ids=[row['sample_id'] for row in rows],features=np.array(values),rules=np.array(rules))
    args.output.with_suffix('.json').write_text(json.dumps({'feature_names':FEATURE_NAMES,'manifest_sha256':hashlib.sha256((args.sample/'manifest.json').read_bytes()).hexdigest()}))

if __name__=='__main__': main()
