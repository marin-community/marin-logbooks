import hashlib
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path

import yaml

ROOT = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
digest = lambda raw: hashlib.sha256(raw).hexdigest()
loader = importlib.util.spec_from_file_location('frozen_primary_reader', '/mnt/storage/karan-scratch/marten-bfcl-run-bootstrap-r3.py')
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)
checked = old.checked_json
primary_path = ROOT / 'bfcl-noise-r1/primary-run-bootstrap-marten-r1.json'
primary_sha = '06f71ba85ac51e23471bba15bcb6198b2a02528f74af302b986fae39c466d9fa'
primary = checked(primary_path, primary_sha)
reference_path = ROOT / 'bfcl-noise-r1/fresh-v01-reference-extension-marten-r1.json'
reference_sha = 'e6a6b5e223acfc945330d9ac5d5bba5a82863bb61e66754048d84526ba5b6df0'
reference = checked(reference_path, reference_sha)
spec_path = ROOT / 'bfcl-noise-r1/A6-run-comparison-extension-spec-marten-r1.json'
spec_sha = 'c2cd0734debd3e6a32274e57a5fae0fceda4dd8e1e120f98313351a4cfec67b2'
spec = checked(spec_path, spec_sha)
assert digest(Path(old.__file__).read_bytes()) == primary['source_script_sha256']
old_index = checked(primary['index_path'], primary['index_sha256'])
old_failures = old.audit_expanded_physical_chain(old_index)
scored, pending, admitted = old.audit(old_index)
assert not pending and admitted == primary['admitted_runs']
arms = {arm: [count for run_id, count in scored[arm]] for arm in old.ARMS}
assert old.bootstrap(arms) == primary['primary_analysis']

def reward(sample):
    values = [value for name, value in sample['metrics'] if name == 'reward']
    assert len(values) == 1
    value = values[0]
    assert type(value) in (int, float) and math.isfinite(value) and value in (0, 1)
    detail = json.loads(sample['grading']['detail'])
    assert detail['scored'] is True and detail['error'] is None
    assert detail['rewards']['reward'] == value == sample['grading']['score']
    return value

reference_index = checked(reference['reference_index_path'], reference['reference_index_sha256'])
assert len(reference_index['physical_reference_runs']) == 9
reference_rows = {row['run_id']: row for row in reference_index['reference_runs']}
task_ids = set()
for row in reference['reference']['runs']:
    entry = reference_rows[row['run_id']]
    for field in ['array', 'report', 'native_record']:
        assert row[field + '_sha256'] == entry[field + '_sha256']
        checked(entry[field], entry[field + '_sha256'])
    samples_path = Path(entry['native_record']).with_name('samples.actual.readonly.json')
    samples = checked(samples_path, row['native_samples_sha256'])
    array = checked(entry['array'], entry['array_sha256'])
    rewards = {sample['doc_id']: reward(sample) for sample in samples}
    assert len(samples) == len(rewards) == 123 and int(sum(rewards.values())) == row['correct']
    assert rewards == {item['doc_id']: item['reward'] for item in array['rows']}
    if task_ids:
        assert set(rewards) == task_ids
    task_ids = set(rewards)
for failure in reference['preserved_partial_reference_workflows']:
    physical = next(row for row in reference_index['physical_reference_runs'] if row['run_id'] == failure['run_id'])
    checked(physical['all_samples'], failure['samples_sha256'])

index_path = ROOT / 'ablation-a6-r1/A6-run-index-20261030T094546.225808Z.json'
index_sha = '0f474b94732882279fe439ee0bcb4a5735aec121507b74f26aa6d85b55309289'
index = checked(index_path, index_sha)
assert index['ALL_FIVE_COMPLETE'] and index['complete123'] == index['required_fresh_runs'] == 5
assert index['physical_workflow_count'] == len(index['physical_A6_runs']) == len(index['A6_runs']) == 5
assert index['primary_analysis_spec_sha256'] == spec_sha
assert index['old26_sha256'] == primary_sha and index['oldfreshref_sha256'] == reference_sha
physical = {row['run_id']: row for row in index['physical_A6_runs']}
assert len(physical) == 5
model_path = ROOT / 'ablation-a6-r1/model-A6.yaml'
model_sha = spec['A6_binding']['model_config_sha256']
assert digest(model_path.read_bytes()) == model_sha
model = yaml.safe_load(model_path.read_bytes())
plain = yaml.safe_load((ROOT / 'bfcl-v01-reference5-r1/model-v01.yaml').read_bytes())
assert {key:value for key,value in model.items() if key not in ['name','location']} == {key:value for key,value in plain.items() if key not in ['name','location']}
native_config_sha = 'ecb4030336060bcf57d9a444805d3e98d6f014a7d9bf0a5778680a60f5b839b7'
readiness_sha = '000610739f5630b79ee649ee8f07c2af35cc7143d74370a5b3b75e19a7c23a4d'
readiness = checked(ROOT / 'ablation-a6-r1/actual-terminal-HF502-audit-retirement-ready-margay-r1.json', readiness_sha)
assert readiness['HF'] == spec['A6_binding']['HF'] and readiness['all502_audit']['frozen_full_fields'] == 104
assert readiness['all502_audit']['trained_master_cast'] == 398
assert readiness['core_event_plan_data_pin_exposure_exact'] and readiness['matching_native_retired_marker_AFTERterminal_export']
for entry in readiness['metadata'].values():
    data = Path(entry['path']).read_bytes()
    assert len(data) == entry['bytes'] and digest(data) == entry['sha256']
a6 = []
configs = []
for row in index['A6_runs']:
    assert {key:value for key,value in row.items() if key not in ['logical_slot','full_physical_chain_retained']} == physical[row['run_id']]
    assert row['state'] == 'COMPLETE123' and row['replacement_number'] == 0 and row['replaces_incomplete_job'] is None
    assert row['full_physical_chain_retained'] == [row['run_id']]
    assert row['HF'] == spec['A6_binding']['HF'] and row['model_config_sha256'] == model_sha
    assert row['method_pin'] == spec['A6_binding']['method_pin']
    launch = checked(row['launch'], row['launch_sha256'])
    assert launch['job'] == row['run_id'] and launch['evaluation_index'] == row['evaluation_index']
    assert launch['HF'] == row['HF'] and launch['model_config_sha256'] == model_sha
    assert launch['method_pin'] == row['method_pin'] and launch['HF_readiness_proof_sha256'] == readiness_sha
    assert launch['primary_statistics_spec_sha256'] == spec_sha and launch['no_P1_prefill'] is True
    native = checked(row['native_record'], row['native_record_sha256'])
    assert native['status'] == 'succeeded' and native['model']['location'] == row['HF']
    assert native['model']['config_digest'] == row['native_model_config_sha256'] == native_config_sha
    coverage = native['coverage']['bfcl_parity']
    assert coverage['n_benchmark'] == coverage['n_attempted'] == coverage['n_scored'] == 123
    assert coverage['n_unanswered'] == 0 and coverage['errors'] == {}
    configs.append(native['model']['config'])
    array = checked(row['array'], row['array_sha256'])
    report = checked(row['report'], row['report_sha256'])
    assert array['score_sha256'] == row['report_sha256']
    assert array['run_id'] == report['run_id'] == row['run_id']
    assert array['results_uri'] == report['results_uri'] == row['results_uri']
    assert len(array['rows']) == 123 and all(set(item) == {'doc_id','reward'} for item in array['rows'])
    rewards = {item['doc_id']:item['reward'] for item in array['rows']}
    assert set(rewards) == task_ids and all(type(value) in (int,float) and math.isfinite(value) and value in (0,1) for value in rewards.values())
    samples_path = Path(row['native_record']).with_name('samples.actual.readonly.json')
    samples_raw = samples_path.read_bytes()
    samples = json.loads(samples_raw)
    assert len(samples) == 123 and {sample['doc_id']:reward(sample) for sample in samples} == rewards
    correct = int(sum(rewards.values()))
    assert correct == row['correct'] == row['score'] == report['correct'] and report['n'] == 123
    assert math.isclose(report['mean_pct'],100*correct/123,rel_tol=0,abs_tol=1e-12)
    assert report['native_record_sha256'] == row['native_record_sha256']
    assert report['source_model_yaml_sha256'] == model_sha and report['native_model_config_sha256'] == native_config_sha
    assert report['method_pin'] == row['method_pin']
    a6.append(dict(run_id=row['run_id'], logical_slot=row['logical_slot'], correct=correct,
        array_sha256=row['array_sha256'], report_sha256=row['report_sha256'],
        native_record_sha256=row['native_record_sha256'], native_samples_sha256=digest(samples_raw)))
assert all(config == configs[0] for config in configs)
reference_config = checked(next(iter(reference_rows.values()))['native_record'], next(iter(reference_rows.values()))['native_record_sha256'])['model']['config']
assert {key:value for key,value in configs[0].items() if key not in ['name','location']} == {key:value for key,value in reference_config.items() if key not in ['name','location']}
a6.sort(key=lambda row:row['run_id'])
assert [row['correct'] for row in a6] == [21,16,17,23,18]
fractions = {arm:[count/123 for count in arms[arm]] for arm in old.ARMS}
generator = random.Random(297)
draws = {arm:[] for arm in old.ARMS}
old_contrasts = {arm:[] for arm in old.ARMS[1:]}
for iteration in range(10000):
    means = {}
    for arm in old.ARMS:
        means[arm] = statistics.fmean(generator.choice(fractions[arm]) for cluster in range(old.EXPECTED[arm]))
        draws[arm].append(means[arm])
    for arm in old.ARMS[1:]:
        old_contrasts[arm].append(means[arm]-means['A0'])
for arm in old.ARMS:
    assert [100*value for value in old.interval(draws[arm])] == primary['primary_analysis']['per_arm'][arm]['unadjusted_mean_CI95_percent']
for arm in old.ARMS[1:]:
    assert [100*value for value in old.interval(old_contrasts[arm])] == primary['primary_analysis']['contrasts'][arm+'_minus_A0']['unadjusted_CI95_pp']
ref_fractions = [row['correct']/123 for row in reference['reference']['runs']]
ref_draws = [statistics.fmean(generator.choice(ref_fractions) for cluster in range(5)) for iteration in range(10000)]
assert [100*value for value in old.interval(ref_draws)] == reference['reference']['unadjusted_mean_CI95_percent']
for arm in old.ARMS:
    assert [100*value for value in old.interval([arm_value-ref_value for arm_value,ref_value in zip(draws[arm],ref_draws,strict=True)])] == reference['arm_minus_new_reference'][arm+'_minus_fresh_v01']['unadjusted_CI95_pp']
a6_fractions = [row['correct']/123 for row in a6]
a6_draws = [statistics.fmean(generator.choice(a6_fractions) for cluster in range(5)) for iteration in range(10000)]
comparators = {'fresh_v01':(ref_fractions,ref_draws),'A3':(fractions['A3'],draws['A3']),'A0':(fractions['A0'],draws['A0'])}
contrasts = {f'A6_minus_{label}':dict(delta_pp=100*(statistics.fmean(a6_fractions)-statistics.fmean(values)),
    unadjusted_CI95_pp=[100*value for value in old.interval([new-old_value for new,old_value in zip(a6_draws,baseline,strict=True)])]) for label,(values,baseline) in comparators.items()}
summary = dict(n=5,runs=a6,success_counts=[row['correct'] for row in a6],
    mean_percent=100*statistics.fmean(a6_fractions),sample_SD_percent=100*statistics.stdev(a6_fractions),
    min_max_percent=[100*min(a6_fractions),100*max(a6_fractions)],
    unadjusted_mean_CI95_percent=[100*value for value in old.interval(a6_draws)])
assert digest(primary_path.read_bytes()) == primary_sha and digest(reference_path.read_bytes()) == reference_sha
output = dict(status='COMPLETE_A6_FROZEN_RUN_BOOTSTRAP_EXTENSION',specification_sha256=spec_sha,
    A6_index_path=str(index_path),A6_index_sha256=index_sha,A6_complete_logical_slots=5,A6_physical_workflows=5,
    A6_replacements=0,A6_complete_admission='All5 full123 native scored binary rewards/sample-record-report-array-launch-config and auditedHF joins PASS',
    frozen26_primary_sha256=primary_sha,frozen_fresh_reference_sha256=reference_sha,
    all_original_and_reference_intervals_reproduced_exactly=True,
    original41_physical_and26_logical_admission_reproduced=True,preserved_original_infra_nonscores=len(old_failures),
    five_fresh_reference_admission_reproduced=True,preserved_partial_reference_runs=4,
    A6=summary,contrasts=contrasts,primary_contrast='A6_minus_fresh_v01',secondary_contrasts=['A6_minus_A3','A6_minus_A0'],
    statistics=spec['bootstrap'],run_order=spec['run_order'],comparison_limits=spec['comparison_limits'],
    current_doses=dict(identity_final=126492,identity_reasoning=55073,replay_final=616153,GLM_ON_encounters=1791,OFF_fallback_ON_encounters=1235,unconsumed_suffix_occurrences=175),
    source_script_sha256=digest(Path(__file__).read_bytes()),historical34_not_pooled=True,old_r5_not_pooled=True,
    partial_rewards_not_imputed_or_pooled=True,model_GPU_remote_calls=0,new_model_training_retry_authority=False)
path = ROOT / 'bfcl-noise-r1/A6-run-bootstrap-extension-marten-r1.json'
assert not path.exists()
path.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(path=str(path),sha256=digest(path.read_bytes()),A6=summary,contrasts=contrasts),indent=2))
