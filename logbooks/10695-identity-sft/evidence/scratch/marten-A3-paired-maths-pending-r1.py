import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


BASE = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
ROOT = BASE / 'A3-paired-math-infra-replacement1-r1'
REFERENCE = BASE / 'A8a-paired-math-r2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    registration = BASE / 'bfcl-noise-r1/A3-A11-paired-maths-preregistration-marten-r1.json'
    assert digest(registration) == 'd9fe50933310d9f49fae86af40bc87e6e8fdb203a418a853c942eaa90b0cb432'
    method_path = BASE / 'bfcl-noise-r1/A8a-paired-math-preregistration-marten-r2.json'
    assert digest(method_path) == 'ff96537805d31ee55f5f5704a7e53139229245c6dd177bac57939ef249eb6c78'
    method = json.loads(method_path.read_text())
    source = Path(method['method']['parent_item_score_and_bootstrap'])
    assert digest(source) == method['method']['parent_functions_sha256']
    module_spec = importlib.util.spec_from_file_location('marten_A3_parent289', source)
    analysis = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(analysis)
    candidate_path = ROOT / 'six-candidate-native-config-coverage-item-arrays-margay-r1.json'
    assert digest(candidate_path) == '824aa6b52dbb36c947b7cd6c0728ac8e06bd2ba42a2db2683a738089b347ee7f'
    reference_path = REFERENCE / 'all12-native-config-coverage-item-arrays-margay-r1.json'
    assert digest(reference_path) == '4ef74452477d3215e0acc532b0d908065db344fbb86f7d4d134fa99b61275527'
    candidate = json.loads(candidate_path.read_text())
    assert candidate['reused_reference_index_sha256'] == digest(reference_path)
    reference_units = [unit for unit in json.loads(reference_path.read_text())['units'] if unit['arm'] == 'v01']
    assert len(candidate['units']) == len(reference_units) == 6
    old_analysis = REFERENCE / 'paired-math-analysis-marten-r1.json'
    assert digest(old_analysis) == 'c410fda8a75cf1ab2b9e2231beec8060062f9d462232e861d53cd8d45925b2df'
    units = candidate['units'] + reference_units
    arrays, configs, coverage = {}, {}, []
    for unit in units:
        arm, trial, task = unit['arm'], unit['trial'], unit['task']
        assert arm in {'A3', 'v01'} and trial in {'greedy', '17', '29'}
        array_path = Path(unit['item_scores']['path'])
        assert array_path.stat().st_size == unit['item_scores']['bytes'] and digest(array_path) == unit['item_scores']['sha256']
        records = json.loads(array_path.read_text())['rows']
        expected = 1319 if task == 'gsm8k_0shot' else 500
        assert len(records) == len({row['doc_id'] for row in records}) == expected
        assert all(type(row['score']) in (int, float) and row['score'] in (0, 1) for row in records)
        assert sum(row['score'] for row in records) == unit['n_correct'] == unit['coverage']['n_correct']
        done_path = Path(unit['native_done_path']) if arm == 'A3' else array_path.with_name('native-done.json')
        assert digest(done_path) == unit['native_done_sha256']
        if arm == 'A3':
            assert done_path.stat().st_size == unit['native_done_bytes']
        done = json.loads(done_path.read_text())
        assert done['outcome']['jobs']['eval'] == unit['actual_eval_job']
        assert done['provenance']['dtype'] == 'bfloat16'
        assert done['outcome']['coverage'][task] == unit['coverage']
        config = done['provenance']['config']
        assert {key: config[key] for key in unit['native_config']} == unit['native_config']
        assert config['chat_template_kwargs']['enable_thinking'] == (task == 'math500')
        assert config['max_gen_toks'] == 8192 and config['max_length'] == 32768 and config['num_concurrent'] == 8
        assert config['apply_chat_template'] is True and config['chat_tokenizer_backend'] == 'none'
        if arm == 'A3':
            configs[(trial, task)] = config
        else:
            assert configs[(trial, task)] == config
        coverage_unit = unit['coverage']
        assert all(coverage_unit[key] == expected for key in ['n_benchmark', 'n_attempted', 'n_scored'])
        metric_key = 'exact_match,flexible-extract' if task == 'gsm8k_0shot' else 'accuracy'
        count_key = 'sample_len' if task == 'gsm8k_0shot' else 'num_total'
        assert unit['metric'] == ('exact_match' if task == 'gsm8k_0shot' else 'accuracy')
        assert unit['filter'] == ('flexible-extract' if task == 'gsm8k_0shot' else 'none')
        average = sum(row['score'] for row in records) / expected
        assert abs(done['outcome']['metrics'][task][metric_key] - average) < 1e-12
        assert abs(done['outcome']['canonical_metrics'][task]['accuracy'] - average) < 1e-12
        assert done['outcome']['metrics'][task][count_key] == expected
        arrays[(arm, trial, task)] = {row['doc_id']: row['score'] for row in records}
        coverage.append(dict(arm=arm, trial=trial, task=task, coverage=coverage_unit, native_done_path=str(done_path), native_done_sha256=digest(done_path), item_scores=unit['item_scores']))
    results = {}
    for task, count in [('gsm8k_0shot', 1319), ('math500', 500)]:
        docs = sorted(arrays[('v01', 'greedy', task)])
        assert len(docs) == count
        assert all(set(arrays[(arm, trial, task)]) == set(docs) for arm in ['A3', 'v01'] for trial in ['greedy', '17', '29'])
        matrices = {arm: np.array([[arrays[(arm, trial, task)][doc] for trial in ['greedy', '17', '29']] for doc in docs]) for arm in ['A3', 'v01']}
        difference = (matrices['A3'] - matrices['v01']).mean(axis=1)
        interval = [float(value * 100) for value in analysis.bootstrap(difference, 10000, 297)]
        aligned = ROOT / f'{task}-aligned-three-decode-marten-r1.npz'
        assert not aligned.exists()
        np.savez(aligned, doc_ids=np.array(docs), A3=matrices['A3'], v01=matrices['v01'], item_mean_difference=difference)
        results[task] = dict(items=count, thinking=task == 'math500', decode_order=['greedy', '17', '29'],
                             counts={arm: values.sum(axis=0).astype(int).tolist() for arm, values in matrices.items()},
                             item_mean_accuracy_pct={arm: float(values.mean() * 100) for arm, values in matrices.items()},
                             A3_minus_existing_fresh_v01=dict(delta_pp=float(difference.mean() * 100), ci95_pp=interval, better=int((difference > 0).sum()), worse=int((difference < 0).sum()), ties=int((difference == 0).sum())),
                             registered_stop_rule='delta_pp < -1.5 AND upper95CI_pp < 0',
                             stop_or_quality_decision=None, reason='PENDING_A3_ACTUAL_SAMPLING_SERVER_HF_TOKENIZER_QUALIFICATION',
                             aligned_arrays=dict(path=str(aligned), sha256=digest(aligned), bytes=aligned.stat().st_size))
    assert digest(old_analysis) == 'c410fda8a75cf1ab2b9e2231beec8060062f9d462232e861d53cd8d45925b2df'
    result = dict(status='LOCAL_ARRAY_AND_NATIVE_DONE_JOINS_PASS_STATISTICS_CONDITIONAL_RUNTIME_QUALIFICATION_PENDING',
                  preregistration_sha256=digest(registration), parent_method_sha256=digest(method_path), candidate_units_index_sha256=digest(candidate_path), reused_reference_index_sha256=digest(reference_path),
                  no_new_reference_serve=True, all10914_binary_item_decode_rows_checked=True,
                  all12_native_done_metric_config_coverage_and_array_joins=True, all6_full_native_configs_equal_reference=True,
                  coverage=coverage, results=results,
                  statistics='Exact hash-bound parent289.bootstrap; within-item3decode mean difference, pairedN-item resamples, NumPydefault_rng297 independently perbenchmark,10000draws/percentile2.5-97.5; no BFCLstream mutation, no task/decode selection or extra contrast',
                  limits=['These statistics are conditional on actual A3 runtime sampling/serverHF/tokenizer qualification; no final STOP/PASS before that read', 'Item arrays are local native-reader projections checked against hashes/coverage/headline metrics; not an independent remote per-item verifier reread', 'Frozen fresh-v01 reference and c410 qualification reused, no historicalsysdefault or newreference mixing', 'First zero-output preempted A3 physical attempt remains separate, no preferred-success selection', 'One checkpoint percondition/fixedpublicitems/three decoding settings, no replicatedtraining or release acceptance'],
                  script_sha256=digest(Path(__file__)), model_GPU_network_storage_calls=0)
    target = ROOT / 'paired-math-statistics-pending-runtime-marten-r1.json'
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'path': str(target), 'sha256': digest(target), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
