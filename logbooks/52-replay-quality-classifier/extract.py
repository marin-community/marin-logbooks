"""Freeze bounded replay windows and a split before annotation or fitting."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials
from tokenizers import Tokenizer

SEED = 20261116
WINDOW = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-frame', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3600)
    args = parser.parse_args()
    configure_coreweave_s3()
    os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY'] = s3_credentials(StoreType.COREWEAVE)
    os.environ['AWS_REGION'] = os.environ['AWS_DEFAULT_REGION'] = data_buckets()['marin-us-east-02a'].signing_region
    args.output.mkdir(parents=True, exist_ok=False)
    seed_bytes = args.seed_frame.read_bytes()
    old = json.loads(seed_bytes)
    strata = {row['cache_dir']: row for row in old}
    frame = [{**row, 'development_seed': True} for row in old]
    seen = {(row['cache_dir'], row['doc_index']) for row in frame}
    rng = np.random.default_rng(SEED)
    counts = {cache: sum(row['cache_dir'] == cache for row in frame) for cache in strata}
    while len(frame) < args.count:
        available = [cache for cache in sorted(strata) if counts[cache] < strata[cache]['rows']]
        minimum = min(counts[cache] for cache in available)
        for cache in [cache for cache in available if counts[cache] == minimum]:
            row = strata[cache]
            index = int(rng.integers(row['rows']))
            while (cache, index) in seen:
                index = int(rng.integers(row['rows']))
            seen.add((cache, index))
            counts[cache] += 1
            frame.append({**row, 'doc_index': index, 'sample_id': len(frame), 'seed': SEED, 'development_seed': False})
            if len(frame) == args.count:
                break
    (args.output / 'frame.json').write_text(json.dumps(frame, indent=2))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    def extract(row):
        destination = args.output / f"sample-{row['sample_id']:04d}.json"
        store = JaggedArrayStore.open(row['cache_dir'] + '/input_ids', mode='r', dtype=np.int32)
        index = row['doc_index']
        start = int(store.offsets[index].read().result()) if index else 0
        end = int(store.offsets[index + 1].read().result())
        length = end - start
        starts = sorted({0, max(0, (length - WINDOW) // 2), max(0, length - WINDOW)})
        # Merge overlapping windows so each visible token occurs only once.
        intervals = []
        for offset in starts:
            stop = min(length, offset + WINDOW)
            if intervals and offset <= intervals[-1][1]:
                intervals[-1][1] = stop
            else:
                intervals.append([offset, stop])
        windows = [np.asarray(store.data[start+a:start+b].read().result()).tolist() for a,b in intervals]
        texts = [tokenizer.decode(window, skip_special_tokens=False) for window in windows]
        score_starts = sorted({0, max(0, (length - 512)//2), max(0, length-512)})
        score_windows = [np.asarray(store.data[start+a:min(end,start+a+512)].read().result()).tolist() for a in score_starts]
        fingerprint = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        draw = int(fingerprint[:8],16) % 10
        split = 'test' if draw < 3 else 'calibration' if draw < 5 else 'train'
        if row['development_seed']:
            split = 'train'
        record = {**row, 'tokens': length, 'population_tokens': row['tokens'], 'population_rows': row['rows'],
                  'windows': texts, 'window_token_intervals': intervals, 'review_clipped': sum(map(len,windows))<length,
                  'text': '\n\n[OMITTED DOCUMENT INTERVAL]\n\n'.join(texts), 'bme_token_ids': score_windows,
                  'bme_window_starts': score_starts, 'text_sha256': fingerprint, 'split': split}
        destination.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return {key: record[key] for key in ('sample_id','cache_dir','doc_index','tokens','window_token_intervals','text_sha256','split','development_seed')}

    with ThreadPoolExecutor(max_workers=8) as pool:
        result = []
        for record in pool.map(extract,frame):
            result.append(record)
            if len(result)%100==0:
                print(json.dumps({'extracted':len(result)}), flush=True)
    # Any repeated visible content touching the existing development set stays in training.
    development_hashes = {row['text_sha256'] for row in result if row['development_seed']}
    for row in result:
        if row['text_sha256'] in development_hashes and row['split'] != 'train':
            row['split'] = 'train'
            path = args.output / f"sample-{row['sample_id']:04d}.json"
            content=json.loads(path.read_text()); content['split']='train'; path.write_text(json.dumps(content,ensure_ascii=False,sort_keys=True))
    manifest = {'seed': SEED, 'seed_frame_sha256':hashlib.sha256(seed_bytes).hexdigest(),
                'tokenizer_sha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                'window_tokens':WINDOW, 'sampling':'balanced unique rows per cache, capped at stratum population; seed400 forced development',
                'splitting':'SHA256 of visible text grouped; first 30% test, next 20% calibration; seed400 and matching content forced training',
                'samples':result}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'completed':len(result),'strata':len(strata)}),flush=True)

if __name__=='__main__':
    main()
