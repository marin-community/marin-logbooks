import ast
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import yaml

root = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
here = root / 'bfcl-noise-r1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    raw = Path(path).read_bytes()
    assert digest(raw) == expected, str(path)
    return json.loads(raw)


spec_sha = 'd6ba2832c39370a53674fd4bfb29b0dc3d9861558b945f69f0af6a11bb161f18'
spec = checked(here / 'A8-run-comparison-preregistration-marten-r1.json', spec_sha)
timing_sha = '27d03583280549aa6ecf90286b8ec9656d35395af6dd08315c55ed73d7304778'
checked(here / 'A8-preregistration-timing-clarification-marten-r1.json', timing_sha)
frozen_paths = {
    'primary': (here / 'primary-run-bootstrap-marten-r1.json', '06f71ba85ac51e23471bba15bcb6198b2a02528f74af302b986fae39c466d9fa'),
    'reference': (here / 'fresh-v01-reference-extension-marten-r1.json', 'e6a6b5e223acfc945330d9ac5d5bba5a82863bb61e66754048d84526ba5b6df0'),
    'A6': (here / 'A6-run-bootstrap-extension-marten-r1.json', '1dbd6ba061ab8525b55ec0b8a7592c652730dac5cc0613846b5c7407309cad3e'),
}
frozen = {label: checked(path, checksum) for label, (path, checksum) in frozen_paths.items()}
replay_path = Path('/mnt/storage/karan-scratch/marten-bfcl-a6-extension-r1.py')
replay_raw = replay_path.read_bytes()
assert digest(replay_raw) == frozen['A6']['source_script_sha256']
module = ast.parse(replay_raw)
output_position = next(position for position, node in enumerate(module.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'output' for target in node.targets))
namespace = {'__name__': 'frozen_a6_admission_and_PRNG_replay', '__file__': str(replay_path)}
exec(compile(ast.Module(body=module.body[:output_position], type_ignores=[]), str(replay_path), 'exec'), namespace)
assert namespace['summary'] == frozen['A6']['A6']
assert namespace['contrasts'] == frozen['A6']['contrasts']
interval = namespace['old'].interval
task_ids = namespace['task_ids']
generator = namespace['generator']
comparators = {
    'fresh_v01': (namespace['ref_fractions'], namespace['ref_draws']),
    'A3': (namespace['fractions']['A3'], namespace['draws']['A3']),
    'A6': (namespace['a6_fractions'], namespace['a6_draws']),
}
baseline_native_config = namespace['configs'][0]
baseline_eval_config = checked(namespace['index']['A6_runs'][0]['native_record'], namespace['index']['A6_runs'][0]['native_record_sha256'])['eval']


def native_scored_reward(sample):
    values = [value for name, value in sample['metrics'] if name == 'reward']
    assert len(values) == 1
    value = values[0]
    assert type(value) in (int, float) and math.isfinite(value) and value in (0, 1)
    detail = json.loads(sample['grading']['detail'])
    assert detail['scored'] is True
    assert detail['rewards']['reward'] == value == sample['grading']['score']
    return value


bindings = {
    'A8a': dict(
        index='A8a/A8a-run-index-20261030T132127.706678Z.json',
        index_sha='dbbef8c94c586a2f27a9b94fb4435b3366db6e2ea2846c38a0301e13b3427eb9',
        yaml_sha='94ef49d52d305933795f1c4c6dfd3ed2baf6ce7f0cbddc545dea664a57940de1',
        native_config_sha='a2c2eb2cf9a34fccac05cce946b0575400a231acf4ee17f07f60fb7981023558',
        readiness_sha='d8b0c26c3bb5a69a92ae270bdfef3c9a663c6b01a3531adb945fac6890896c4d',
        identity_R=5388, trace_occurrences=179, padding_only=7, final_update_nonpadding=1,
    ),
    'A8b': dict(
        index='A8b/A8b-run-index-20261030T113535.438900Z.json',
        index_sha='48bfaa33dae7d5180d107308bdf0257363e980d47153c90802874cf775d09c11',
        yaml_sha='e2f26c1673a23d721f8ace4e5d947ec4f48eea87106895a46f32630bb56c63eb',
        native_config_sha='93ffebd1fbbe627435a82a7a7bff534f6576e32a8d056f4a6491d5d37d48e6b9',
        readiness_sha='37ec4fe83b38c78f8b46cf8af31baf4c8c18072bbb6e176a81fa85d4fbcac87d',
        identity_R=16292, trace_occurrences=537, padding_only=6, final_update_nonpadding=2,
    ),
}
admitted = {}
for arm, binding in bindings.items():
    index_path = root / 'ablation-a8-r1' / binding['index']
    index = checked(index_path, binding['index_sha'])
    assert index['ALL_FIVE_COMPLETE'] and index['complete123'] == index['required_fresh_runs'] == 5
    assert index['physical_workflow_count'] == len(index['physical_runs']) == len(index['runs']) == 5
    assert index['primary_analysis_spec_sha256'] == spec_sha
    assert index['frozen_A6_analysis_sha256'] == frozen_paths['A6'][1]
    assert index['old26_sha256'] == frozen_paths['primary'][1] and index['oldfreshref_sha256'] == frozen_paths['reference'][1]
    physical = {row['run_id']: row for row in index['physical_runs']}
    assert len(physical) == 5
    assert {row['evaluation_index'] for row in index['runs']} == set(range(1, 6))
    model_path = root / f'ablation-a8-r1/model-{arm}.yaml'
    assert digest(model_path.read_bytes()) == binding['yaml_sha']
    model = yaml.safe_load(model_path.read_bytes())
    plain = yaml.safe_load((root / 'bfcl-v01-reference5-r1/model-v01.yaml').read_bytes())
    assert {key: value for key, value in model.items() if key not in ['name', 'location']} == {key: value for key, value in plain.items() if key not in ['name', 'location']}
    HF = f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-4bfe87adbb03/{arm}/run/hf/updates-36'
    readiness_path = root / f'ablation-a8-r1/{arm}/actual-audited-HF-ready-margay-r1.json'
    readiness = checked(readiness_path, binding['readiness_sha'])
    assert readiness['HF'] == HF and readiness['all502_audit_and_exact_event_retirement_ready_markers_PASS']
    assert readiness['all502_audit']['frozen_full_fields'] == 104 and readiness['all502_audit']['trained_master_cast'] == 398
    assert readiness['all36_ordered_finite_training_updates'] and readiness['frozen_special_row_hashes_exact_A6']
    for entry in readiness['metadata'].values():
        raw = Path(entry['path']).read_bytes()
        assert len(raw) == entry['bytes'] and digest(raw) == entry['sha256']
    admitted[arm] = []
    for row in index['runs']:
        assert {key: value for key, value in row.items() if key not in ['logical_slot', 'full_physical_chain_retained']} == physical[row['run_id']]
        assert row['state'] == 'COMPLETE123' and row['replacement_number'] == 0 and row['replaces_incomplete_job'] is None
        assert row['full_physical_chain_retained'] == [row['run_id']]
        assert row['HF'] == HF and row['model_config_sha256'] == binding['yaml_sha']
        assert row['method_pin'] == 'caba0fa38b780af42da54af2f188ba5bfcc28501'
        launch = checked(row['launch'], row['launch_sha256'])
        assert launch['job'] == row['run_id'] and launch['evaluation_index'] == row['evaluation_index']
        assert launch['HF'] == HF and launch['model_config_sha256'] == binding['yaml_sha']
        assert launch['method_pin'] == row['method_pin'] and launch['HF_readiness_proof_sha256'] == binding['readiness_sha']
        assert launch['primary_statistics_spec_sha256'] == spec_sha and launch['no_P1_prefill'] is True
        assert launch['application_submit_calls'] == 1 and launch['replacement_number'] == 0
        assert launch['event_sha256'] == readiness['metadata']['event.json']['sha256']
        assert launch['audit_sha256'] == readiness['metadata']['audit.json']['sha256']
        native = checked(row['native_record'], row['native_record_sha256'])
        assert native['status'] == 'succeeded' and native['model']['location'] == HF
        assert native['model']['config_digest'] == row['native_model_config_sha256'] == binding['native_config_sha']
        assert {key: value for key, value in native['model']['config'].items() if key not in ['name', 'location']} == {key: value for key, value in baseline_native_config.items() if key not in ['name', 'location']}
        assert native['eval'] == baseline_eval_config
        coverage = native['coverage']['bfcl_parity']
        assert coverage['n_benchmark'] == coverage['n_attempted'] == coverage['n_scored'] == 123
        assert coverage['n_unanswered'] == 0
        array = checked(row['array'], row['array_sha256'])
        report = checked(row['report'], row['report_sha256'])
        assert array['score_sha256'] == row['report_sha256']
        assert array['run_id'] == report['run_id'] == row['run_id']
        assert array['results_uri'] == report['results_uri'] == row['results_uri']
        assert len(array['rows']) == 123 and all(set(item) == {'doc_id', 'reward'} for item in array['rows'])
        rewards = {item['doc_id']: item['reward'] for item in array['rows']}
        assert set(rewards) == task_ids and all(type(value) in (int, float) and math.isfinite(value) and value in (0, 1) for value in rewards.values())
        samples_raw = Path(row['native_record']).with_name('samples.actual.readonly.json').read_bytes()
        samples = json.loads(samples_raw)
        assert len(samples) == 123 and {sample['doc_id']: native_scored_reward(sample) for sample in samples} == rewards
        scored_errors = [dict(doc_id=sample['doc_id'], reward=rewards[sample['doc_id']],
            error=json.loads(sample['grading']['detail'])['error']) for sample in samples if json.loads(sample['grading']['detail'])['error'] is not None]
        assert dict(Counter(item['error']['type'] for item in scored_errors)) == coverage['errors']
        correct = int(sum(rewards.values()))
        assert correct == row['correct'] == row['score'] == report['correct'] and report['n'] == 123
        assert math.isclose(report['mean_pct'], 100 * correct / 123, rel_tol=0, abs_tol=1e-12)
        assert report['native_record_sha256'] == row['native_record_sha256']
        assert report['source_model_yaml_sha256'] == binding['yaml_sha'] and report['native_model_config_sha256'] == binding['native_config_sha']
        assert report['method_pin'] == row['method_pin']
        admitted[arm].append(dict(
            run_id=row['run_id'], logical_slot=row['logical_slot'], correct=correct,
            array_sha256=row['array_sha256'], report_sha256=row['report_sha256'],
            native_record_sha256=row['native_record_sha256'], native_samples_sha256=digest(samples_raw),
            launch_sha256=row['launch_sha256'],
            native_scored_errors=scored_errors,
        ))
    admitted[arm].sort(key=lambda row: row['run_id'])
fractions = {arm: [row['correct'] / 123 for row in admitted[arm]] for arm in bindings}
draws = {arm: [] for arm in bindings}
for iteration in range(10000):
    for arm in ['A8a', 'A8b']:
        draws[arm].append(statistics.fmean(generator.choice(fractions[arm]) for cluster in range(5)))
summaries = {arm: dict(
    n=5, runs=admitted[arm], success_counts=[row['correct'] for row in admitted[arm]],
    mean_percent=100 * statistics.fmean(fractions[arm]), sample_SD_percent=100 * statistics.stdev(fractions[arm]),
    min_max_percent=[100 * min(fractions[arm]), 100 * max(fractions[arm])],
    unadjusted_mean_CI95_percent=[100 * value for value in interval(draws[arm])],
) for arm in bindings}
contrasts = {f'{arm}_minus_{label}': dict(
    delta_pp=100 * (statistics.fmean(fractions[arm]) - statistics.fmean(values)),
    unadjusted_CI95_pp=[100 * value for value in interval([new - old for new, old in zip(draws[arm], comparator_draws, strict=True)])],
) for arm in bindings for label, (values, comparator_draws) in comparators.items()}
for path, checksum in frozen_paths.values():
    assert digest(path.read_bytes()) == checksum
output = dict(
    status='COMPLETE_JOINT_A8_FROZEN_RUN_BOOTSTRAP_EXTENSION', specification_sha256=spec_sha,
    timing_clarification_sha256=timing_sha, admitted_runs=admitted,
    new_complete_logical_slots=10, new_physical_workflows=10, new_replacements=0,
    complete_admission='All10 full123 native binary reward/sample/record/report/array/launch/config/evalconfig and502auditedHF joinsPASS',
    frozen_result_hashes={label: checksum for label, (path, checksum) in frozen_paths.items()},
    all_06f71_e6a6_1dbd_mean_and_contrast_intervals_reproduced_exactly=True,
    exact_A6_replay_script_sha256=digest(replay_raw),
    original41_physical26_scored15_infra_nonscores_reaudited=True,
    fresh_reference9_physical5_scored4_partial_infrastructure_workflows_reaudited=True,
    A6_five_complete_runs_readmitted=True, summaries=summaries, contrasts=contrasts,
    primary_contrasts=['A8a_minus_fresh_v01', 'A8b_minus_fresh_v01'],
    secondary_contrasts=['A8a_minus_A3', 'A8a_minus_A6', 'A8b_minus_A3', 'A8b_minus_A6'],
    statistics=spec['analysis'], interpretation_limits=spec['interpretation_limits'],
    new_indexes={arm: dict(path=str(root / 'ablation-a8-r1' / binding['index']), sha256=binding['index_sha']) for arm, binding in bindings.items()},
    current_doses={arm: dict(identity_final=126492, replay_final=616153, identity_reasoning=binding['identity_R'],
        GLM_ON_encounters=binding['trace_occurrences'], eligible_traced_ON_encounters=1791,
        consumed_occurrences=7266, padding_only_sequences=binding['padding_only'], final_update_nonpadding_sequences=binding['final_update_nonpadding']) for arm, binding in bindings.items()},
    no_outcome_added_A8a_A8b_interval=True, no_task_or_repeat_pairing=True,
    historical34_and_old_r5_not_pooled=True, no_partial_rewards_imputed=True,
    model_GPU_remote_calls=0, new_training_model_retry_or_release_authority=False,
    source_script_sha256=digest(Path(__file__).read_bytes()),
)
path = here / 'A8-joint-run-bootstrap-extension-marten-r1.json'
assert not path.exists()
path.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(dict(path=str(path), sha256=digest(path.read_bytes()), summaries=summaries, contrasts=contrasts), indent=2))
