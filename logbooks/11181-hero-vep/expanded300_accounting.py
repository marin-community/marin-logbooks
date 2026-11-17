"""Summarize bounded scoring attempts without publishing wire responses."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import sys

p = Path(sys.argv[1])
labels = ['compatibility'] + [f'chunk-{i:02d}' for i in range(10)] + ['chunk-02-recovery']
state = json.loads((p/'execution-state-private.json').read_text())
assert state['completed_chunks'] == 10 and not state['invalid']
statuses = collections.Counter()
routes = collections.Counter()
usage = collections.Counter()
record_types = collections.Counter()
fingerprints = set()
scores = []
repeats = []
attempts = {}
artifact_hashes = {}
for label in labels:
    attempts[label] = json.loads((p/f'{label}-summary.json').read_text())
    assert attempts[label]['completed'] == (label != 'chunk-02')
    with (p/f'{label}-wire-private.jsonl').open() as source:
        for line in source:
            r = json.loads(line)
            statuses[r['status']] += 1
            routes[r['route']] += 1
            if r['route'] == 'v1/completions' and r['status'] == 200:
                response = json.loads(r['response_text'])
                fingerprints.add(response['system_fingerprint'])
                for key, value in response['usage'].items():
                    if isinstance(value, (int,float)):
                        usage[key] += value
    path = p/f'{label}-scores.jsonl'
    artifact_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        record_types[r['type']] += 1
        if r['type'] == 'score': scores.append(r)
        if r['type'] == 'repeat': repeats.append(r)
assert len(scores) == len({r['index'] for r in scores}) == 4000
assert len(fingerprints) == 1
assert len(repeats) == 252 and max(r['delta'] for r in repeats) <= 1e-6
assert sum(statuses.values()) == state['requests']
assert statuses[502] == 1 and len(statuses) == 2
assert routes['v1/completions'] == 4285
elapsed = attempts['chunk-09']['total_elapsed']
result = dict(complete=True, split='train', variants=600, subsets=2,
    fresh_targets=4000, reused_targets=800, total_panel_targets=4800,
    compatibility_comparisons=32, repeat_checks=252,
    maximum_repeat_delta=max(r['delta'] for r in repeats),
    successful_completions=4284, failed_completions=1,
    service_http=dict(statuses), service_http_routes=dict(routes),
    extra_model_discovery_checks=2, extra_checks_description='One initial availability preflight and one post-502 availability check; both HTTP200.',
    usage=dict(usage), same_serving_fingerprint=True,
    reuse=state['reuse'], total_elapsed_seconds=elapsed,
    started_utc=datetime.datetime.fromtimestamp(state['start_epoch'],datetime.timezone.utc).isoformat(),
    completed_utc=datetime.datetime.fromtimestamp(state['start_epoch']+elapsed,datetime.timezone.utc).isoformat(),
    max_completion_latency=max(x['max_completion_latency'] for x in attempts.values()),
    attempts=attempts, score_artifact_sha256=artifact_hashes,
    frozen_metrics_sha256=hashlib.sha256((p/'expanded300_metrics.py').read_bytes()).hexdigest())
(p/'expanded300-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['attempts','score_artifact_sha256']},indent=2))
