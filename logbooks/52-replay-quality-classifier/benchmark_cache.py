"""Measure the deployed scorer against real bounded cache reads, without text output."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials

from marin.datakit.low_value.model import load_model, load_operating_point
from marin.datakit.low_value.scoring import load_scorer, score_token_cache


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.root
    hashes=json.loads((args.bundle/'SHA256SUMS.json').read_text())
    model=load_model(str(args.bundle/'model.json'),hashes['model.json'])
    point=load_operating_point(str(args.bundle/'calibration.json'),hashes['calibration.json'],model.sha256,.05)
    scorer=load_scorer(model)
    configure_coreweave_s3()
    os.environ['AWS_ACCESS_KEY_ID'],os.environ['AWS_SECRET_ACCESS_KEY']=s3_credentials(StoreType.COREWEAVE)
    os.environ['AWS_REGION']=os.environ['AWS_DEFAULT_REGION']=data_buckets()['marin-us-east-02a'].signing_region
    frame=json.loads((root/'sample-v1/frame.json').read_text())[:100]
    start=time.monotonic()
    def one(row):
        result=list(score_token_cache(row['cache_dir'],scorer=scorer,operating_point=point,indices=[row['doc_index']]))[0]
        return {'sample_id':row['sample_id'],'doc_index':row['doc_index'],'tokens':result.tokens,
                'low_value_score':result.low_value_score,'is_low_value':result.is_low_value}
    with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(one,frame))
    elapsed=time.monotonic()-start
    expected=np.load(root/'final-v1/scores.npz')['score'][:100]
    actual=np.array([row['low_value_score'] for row in results])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
    artifact={'documents':len(results),'wall_seconds':elapsed,'workers':8,'documents_per_second':len(results)/elapsed,
              'scope':'cold independent cache rows on workstation; includes network/metadata; not a fleet throughput estimate',
              'max_probability_error':float(np.max(np.abs(actual-expected))),'rows':results}
    args.output.write_text(json.dumps(artifact,indent=2))
    print(json.dumps({k:v for k,v in artifact.items() if k!='rows'},indent=2))

if __name__=='__main__': main()
