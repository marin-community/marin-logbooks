"""Native matched-group metrics for the frozen expanded train panels."""
import hashlib
import json
import sys
from pathlib import Path

import marin_dna_evals.metrics as native
import numpy as np
import pandas as pd


def metrics(rows):
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == 'bee720324d61ea8b88a4644b6bd036569e9bab6f8e931d6335d165fb2e2b2395'
    frame = pd.DataFrame(rows)
    assert len(frame) == 600
    result = {}
    for subset, panel in frame.groupby('subset'):
        groups = panel.groupby('match_group')
        assert len(panel) == 300 and panel.id.is_unique and panel.match_group.nunique() == 30
        assert (groups.size() == 10).all() and (groups.label.sum() == 1).all()
        assert panel.label.isin([True, False]).all()
        assert np.isfinite(panel[['null', 'healthy']].to_numpy()).all()
        record = dict(n_variants=300, n_groups=30, positives=30, exploratory=True,
                      bootstrap_replicates=2000, bootstrap_seed=11181,
                      auprc={}, within_group_pairwise_accuracy={})
        for method in ['null', 'healthy']:
            record['auprc'][method] = native.auprc_with_bootstrap_se(
                panel.label, panel[method], panel.match_group, n_bootstrap=2000, rng=11181)
            wins = []
            for _, group in groups:
                positive = group.loc[group.label.astype(bool), method].iloc[0]
                negatives = group.loc[~group.label.astype(bool), method]
                wins.extend((positive > negatives).astype(float) + 0.5 * (positive == negatives))
            record['within_group_pairwise_accuracy'][method] = float(np.mean(wins))
        record['healthy_minus_null'] = native.paired_metric_delta_bootstrap(
            panel.label, panel.healthy, panel['null'], panel.match_group, n_bootstrap=2000, rng=11181)
        result[subset] = record
    assert len(result) == 2
    return result


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(json.dumps(metrics(json.loads(Path(sys.argv[1]).read_text())), indent=2) + '\n')
