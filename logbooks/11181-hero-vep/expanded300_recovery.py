"""One explicit recovery of a transport-interrupted expanded chunk."""
import json
from expand_likelihood import chunk


def recover(client, plan, indices, anchors, prior_path, emit):
    prior = [json.loads(line) for line in prior_path.read_text().splitlines()]
    assert prior[0]['type'] == 'start' and prior[0]['indices'] == indices
    assert not any(row['type'] == 'complete' for row in prior)
    scores = [row for row in prior if row['type'] == 'score']
    assert [row['index'] for row in scores] == indices[:len(scores)]
    assert 0 < len(scores) < len(indices)
    for row in prior:
        if row['type'] == 'repeat':
            assert row['delta'] <= 1e-6
        if row['type'] == 'score':
            task = plan['tasks'][row['index']]
            assert all(row[key] == task[key] for key in ['subset', 'variant', 'strand', 'method', 'allele', 'input_sha256'])
    emit(dict(type='recovery', retained_targets=len(scores), remaining_targets=len(indices)-len(scores)))
    # chunk rechecks every anchor before any new target, then every 80 targets.
    chunk(client, plan, indices[len(scores):], anchors, emit)
