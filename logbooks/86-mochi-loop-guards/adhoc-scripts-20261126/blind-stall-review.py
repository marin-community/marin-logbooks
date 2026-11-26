"""Show registered stall-review evidence without verdicts, future turns or cut lengths."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('sample')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--count', type=int, default=5)
args = parser.parse_args()
document = json.loads(Path(args.sample).read_text())
for index, trial in enumerate(document['trials'][args.start:args.start+args.count], args.start):
    fire = trial['fire_index']
    window = {turn['index']: turn for turn in trial['window']}
    completed = [window[i] for i in range(fire-2,fire+1)]
    record = {'review_index':index,'sample_rank':trial['sample_rank'],'cohort':trial['cohort'], 'wait_flags':[turn['wait'] for turn in completed], 'action_sha256':[turn['action_sha256'] for turn in completed], 'observation_sha256':[turn['observation_sha256'] for turn in completed], 'repeated_action':completed[-1]['action'][:1600], 'repeated_observation':completed[-1]['observation'][:2400], 'assistant_intent':[turn['message'][:1000] for turn in completed]}
    if fire-3 in window:
        previous = window[fire-3]
        record['preceding_context'] = {k:previous[k][:900] for k in ('action','observation','message')}
    print(json.dumps(record,indent=2))
