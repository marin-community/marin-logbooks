"""Preserve the published rule's separate machine-junk priority in its ranking."""
import argparse
import json
from pathlib import Path

import numpy as np
from baseline_rules import JUNK_CHUNK_SHARE, MACHINE_JUNK, MACHINE_JUNK_CHUNK_SHARE, junk_verdict
from features import SPECIAL


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root
    ids=[]; scores=[]
    for path in sorted((root/'sample-v1').glob('sample-*.json')):
        row=json.loads(path.read_text())
        verdict=junk_verdict([SPECIAL.sub('',text) for text in row['windows']])
        machine=sum(count for category,count in verdict.counts.items() if category in MACHINE_JUNK)
        score=max(verdict.junk_share/JUNK_CHUNK_SHARE,machine/max(1,verdict.chunks)/MACHINE_JUNK_CHUNK_SHARE)
        assert (score>=1)==(verdict.category is not None)
        ids.append(row['sample_id']); scores.append(score)
    np.savez_compressed(root/'native-rules-v1.npz',ids=ids,score=scores)

if __name__=='__main__': main()
