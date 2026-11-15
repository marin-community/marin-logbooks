"""Compute bounded-window Harrier features only when no document-ID join exists."""
import argparse
import hashlib
import io
import json

import numpy as np
from rigging.filesystem.factory import open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from transformers import AutoTokenizer

from marin.processing.embeddings.harrier import HARRIER_DIM, HARRIER_REPO, HARRIER_REVISION, HarrierConfig, load_harrier

MAX_TOKENS=1024


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    configure_coreweave_s3()
    with open_url(args.input,'rb') as stream:
        contents=stream.read()
    rows=[json.loads(line) for line in contents.splitlines()]
    config=HarrierConfig(model=HARRIER_REPO,revision=HARRIER_REVISION,tokenizer_revision=HARRIER_REVISION,
                         dimensions=HARRIER_DIM,max_tokens=MAX_TOKENS,compute_dtype='bfloat16',device='cuda',
                         batch_size=16,batch_token_limit=16384)
    tokenizer=AutoTokenizer.from_pretrained(HARRIER_REPO,revision=HARRIER_REVISION)
    embedder=load_harrier(config)
    vectors=[]
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]
        texts=[]
        spans=[]
        for row in batch:
            begin=len(texts)
            for text in row['windows']:
                tokens=tokenizer.encode(text,add_special_tokens=False)
                # Retokenization can add tokens at the decode boundary; leave margin.
                if len(tokens)>MAX_TOKENS-8:
                    text=tokenizer.decode(tokens[:MAX_TOKENS-8],skip_special_tokens=True)
                while len(tokenizer.encode(text))>MAX_TOKENS:
                    text=text[:-16]
                texts.append(text)
            spans.append((begin,len(texts)))
        embedding=embedder.embed(texts)
        pooled=np.stack([embedding[a:b].mean(axis=0) for a,b in spans])
        pooled/=np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
        vectors.append(pooled)
        print(json.dumps({'embedded':min(start+32,len(rows))}),flush=True)
    buffer=io.BytesIO()
    np.savez_compressed(buffer,ids=np.array([row['sample_id'] for row in rows]),embedding=np.concatenate(vectors))
    with open_url(args.output+'.npz','wb') as stream:
        stream.write(buffer.getvalue())
    with open_url(args.output+'.json','w') as stream:
        json.dump({'config':config.__dict__,'input_sha256':hashlib.sha256(contents).hexdigest(),'output_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),'pooling':'mean of L2-normalized head/middle/tail window embeddings, renormalized; <=1024 Harrier tokens each'},stream)

if __name__=='__main__':
    main()
