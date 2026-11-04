import ast
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import pyarrow.ipc as ipc


ROOT = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
HERE = ROOT / 'bfcl-noise-r1'
INDEX_SHA = '1c5838712e6fe51929463275150f6378f0af68ef246cd0623860e796ead19a42'
SPEC_SHA = 'd3c966d90391dd7e1fabb5e6e29ecdb3382f454c71fd3a4a312878d03cea8738'
HF = 's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-a12-export-recovery-r1-20261031/A12/run/hf/updates-18'
READINESS_SHA = '39dea9c2f14f6d082af9649aa0398453ed07f7add4b1e1b8557c430a507166a5'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected, length=None):
    raw = Path(path).read_bytes()
    assert digest(raw) == expected, str(path)
    assert length is None or len(raw) == length, str(path)
    return json.loads(raw)


def descriptor(entry):
    return checked(entry['path'], entry['sha256'], entry['bytes'])


spec = checked(HERE / 'A12-A13-science-and-analysis-preregistration-marten-r1.json', SPEC_SHA)
joint = checked(HERE / 'A11-minus-A9-registered-contrast-final-marten-r1.json', '4b22c9cd2cde36b57733ead5da59a9d10ec040c1ba89c8a72de895921b2cfc29')
parent = Path('/mnt/storage/karan-scratch/marten-bfcl-A11-minus-A9-final-r1.py')
raw = parent.read_bytes()
assert digest(raw) == joint['source_script_sha256']
module = ast.parse(raw)
position = next(position for position, node in enumerate(module.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'output' for target in node.targets))
prior = {'__file__': str(parent), '__name__': 'frozen_prefix_through_A11'}
exec(compile(ast.Module(body=module.body[:position], type_ignores=[]), str(parent), 'exec'), prior)
assert prior['contrast'] == joint['contrast']['A11_minus_A9']
canonical = prior['canonical']
interval = prior['interval']
start_state = canonical.getstate()
start_sha = digest(json.dumps(start_state).encode())
assert start_sha == '0b389af49c68f2e9368db6e6419e5829365848133316ab47ea1b8ef03ff6eba3'
assert start_sha in spec['BFCL_statistics']['stream']
task_ids = prior['namespace']['task_ids']
baseline_model = prior['namespace']['baseline_native_config']
baseline_eval = prior['namespace']['baseline_eval_config']
index_path = ROOT / 'A12-BFCL-production-five-r1/authorized-nine-physical-and-admitted-five-logical-BFCL-index-margay-r1.json'
index = checked(index_path, INDEX_SHA)
assert index['state'] == 'ALL5_LOGICAL_BFCL_CHAINS_FULL123_NO_PENDING_NO_EXHAUSTION'
assert index['planned_logical_runs'] == index['full123_logical_runs'] == len(index['logical']) == 5
assert index['physical_runs'] == len(index['physical']) == 9
assert index['all_physical_attempts_closed'] and index['missing_windows'] == 0
assert index['owned_jobs'] == index['unique_physical_attempt_UIDs'] == 18
assert len({row['job'] for row in index['physical']}) == 9
physical = {}
admission = []
for entry in index['physical']:
    proof_path = Path(entry['proof'])
    proof = checked(proof_path, entry['proof_sha256'])
    launch = checked(proof_path.with_name('launch.json'), proof['launch_sha256'])
    assert launch['job'] == launch['expected_job'] == proof['job'] == entry['job']
    assert launch['application_submit_calls'] == 1 and launch['source_pin'] == '43bba4a16d175072b20fea27b8fb5524c377c79f'
    assert launch['local_native_workspace_bundle_sha256'] == '5912c661b8eee8899804f97cf2b18c4bd2bf44321645bff70413db0667a85b7b'
    assert launch['HF'] == proof['HF'] == proof['source_HF'] == HF
    assert launch['HF_readiness_proof_sha256'] == proof['readiness_sha256'] == READINESS_SHA
    assert proof['SDK_terminal'] and proof['SDK_state'] == 'succeeded'
    assert proof['original_slot'] == entry['original_slot'] and proof['replacement_number'] == entry['replacement_number']
    native = descriptor(proof['native_record_full_original_restricted_local'])
    assert native['status'] == 'succeeded' and native['run_id'] == proof['native_eval_id'] == entry['native_eval_id']
    assert native['results_path'] == proof['results_uri'] == entry['results_uri']
    assert native['model']['location'] == HF
    assert native['model']['config_digest'] == proof['native_model_config_sha256']
    assert {key: value for key, value in native['model']['config'].items() if key not in ['name', 'location']} == {key: value for key, value in baseline_model.items() if key not in ['name', 'location']}
    assert native['eval'] == baseline_eval
    samples = descriptor(proof['samples_full_JSON_native_reader_projection'])
    arrow = proof['samples_full_typed_native_reader_projection']
    arrow_raw = Path(arrow['path']).read_bytes()
    assert digest(arrow_raw) == arrow['sha256'] and len(arrow_raw) == arrow['bytes']
    table = ipc.open_file(arrow['path']).read_all()
    assert table.num_rows == 123 and len(table.column_names) == 23
    assert json.loads(json.dumps(table.to_pylist())) == samples
    assert len(samples) == len({row['doc_id'] for row in samples}) == 123
    assert {row['doc_id'] for row in samples} == task_ids
    rewards = {}
    errors = []
    scored = 0
    for sample in samples:
        detail = json.loads(sample['grading']['detail'])
        values = [value for name, value in sample['metrics'] if name == 'reward']
        if detail['scored']:
            assert len(values) == 1
            reward = values[0]
            assert type(reward) in (int, float) and math.isfinite(reward) and reward in (0, 1)
            assert detail['rewards']['reward'] == sample['grading']['score'] == reward
            scored += 1
        else:
            assert not values and sample['grading']['score'] is None and sample['grading']['passed'] is None
            assert detail['error']['category'] == 'infrastructure'
            reward = None
        rewards[sample['doc_id']] = reward
        if detail['error'] is not None:
            errors.append(dict(doc_id=sample['doc_id'], scored=detail['scored'], reward=reward, type=detail['error']['type'], category=detail['error']['category']))
    projected = {row['doc_id']: row['reward'] for row in proof['rows']}
    assert len(proof['rows']) == 123 and rewards == projected
    coverage = native['coverage']['bfcl_parity']
    assert coverage == proof['native_coverage']
    assert coverage['n_benchmark'] == coverage['n_attempted'] == 123
    assert scored == coverage['n_scored'] == proof['scored_binary_rewards'] == entry['scored_binary_rewards']
    assert dict(Counter(error['type'] for error in errors)) == coverage['errors']
    observed = int(sum(value for value in rewards.values() if value is not None))
    assert observed == proof['observed_correct'] == native['metrics']['bfcl_parity']['solved']
    valid = scored == 123
    assert valid == proof['logical_full123_admissible'] == entry['primary_admissible']
    assert (observed if valid else None) == proof['score'] == entry['score']
    physical[entry['job']] = dict(slot=entry['original_slot'], replacement_number=entry['replacement_number'], job=entry['job'], correct=observed if valid else None, observed_correct=observed, scored=scored, proof_sha256=entry['proof_sha256'], native_record_sha256=proof['native_record_sha256'], JSON_sha256=proof['samples_full_JSON_native_reader_projection']['sha256'], typed_reader_sha256=arrow['sha256'], launch_sha256=proof['launch_sha256'], native_scored_and_unscored_errors=errors, primary_admissible=valid)
    if valid:
        admission.append(physical[entry['job']])
for entry in index['logical']:
    chain = sorted([row for row in physical.values() if row['slot'] == entry['original_slot']], key=lambda row: row['replacement_number'])
    assert [row['replacement_number'] for row in chain] == list(range(len(chain)))
    assert len(chain) == (1 if entry['original_slot'] == 4 else 2)
    assert all(not row['primary_admissible'] for row in chain[:-1])
    assert chain[-1]['job'] == entry['job'] and chain[-1]['primary_admissible']
assert len(admission) == 5
admission.sort(key=lambda row: row['job'])
fractions = [row['correct'] / 123 for row in admission]
generator = random.Random()
generator.setstate(start_state)
draws = [statistics.fmean(generator.choice(fractions) for cluster in range(5)) for iteration in range(10000)]
for iteration in range(10000):
    for cluster in range(5):
        canonical.choice(range(5))
assert generator.getstate() == canonical.getstate()
summary = dict(planned=5, n=5, exhausted_infra_no_score=0, pending=0, physical_workflows=9, owned_jobs=18, unique_attempts=18, missing_windows=0, runs=admission, success_counts_original_slot_order=[row['score'] for row in sorted(index['logical'], key=lambda row: row['original_slot'])], mean_percent=100 * statistics.fmean(fractions), sample_SD_percent=100 * statistics.stdev(fractions), min_max_percent=[100 * min(fractions), 100 * max(fractions)], unadjusted_mean_CI95_percent=[100 * value for value in interval(draws)])
contrasts = {}
for label in ['fresh_v01', 'A8a']:
    values, comparator_draws = prior['comparators'][label]
    contrasts['A12_minus_' + label] = dict(delta_pp=100 * (statistics.fmean(fractions) - statistics.fmean(values)), unadjusted_CI95_pp=[100 * value for value in interval([new - old for new, old in zip(draws, comparator_draws, strict=True)])])
for name, expected in prior['FROZEN'].items():
    assert digest((HERE / name).read_bytes()) == expected
output = dict(
    status='FINAL_A12_FIVE_FULL123_REGISTERED_RUN_CLUSTER_BOOTSTRAP',
    preregistration_sha256=SPEC_SHA, reduced_n_addendum_sha256=prior['amendment_sha'],
    index_path=str(index_path), index_sha256=INDEX_SHA, actual_HF=HF, actual_HF_readiness_sha256=READINESS_SHA,
    native_qualification=dict(all9_full_native_records_and_1107_full_sample_rows_checked=True, final615_all_grading_scored_binary=True, full_JSON_equals_23column_typed_reader_projection=True, exact_fixed123_docIDs_each_physical=True, full_native_model_profile_except_name_location_and_evalconfig_match_frozen_A6=True, source43b_native5912_launch_and_own_HF_metadata_joins=True, native_scored_agent_failures_retained=True, original7_infrastructure_missing_rows_NULL_not_zero=True),
    all9_physical_proofs=list(physical.values()), summary=summary, contrasts=contrasts,
    primary=['A12_minus_fresh_v01'], secondary=['A12_minus_A8a'],
    A12_block_start_state_sha256=start_sha,
    A13_canonical_block_start_state_sha256=digest(json.dumps(canonical.getstate()).encode()),
    A12_shared_draws_sha256=digest(json.dumps(draws).encode()),
    exact_old26_reference_A6_A8_A9_A10_A11_and_registered_A11_minus_A9_intervals_replayed=True,
    frozen_analysis_files_unchanged={**prior['FROZEN'], 'A11-minus-A9-registered-contrast-final-marten-r1.json': '4b22c9cd2cde36b57733ead5da59a9d10ec040c1ba89c8a72de895921b2cfc29'},
    canonical_actual_n5_states_equal=True, sibling_outcome_wait=False,
    statistics='Frozen PythonRandom297 stream through old26/ref/A6/A8/A9/canonicalA10n5/A11, then10000 A12 n5 draws from lexical actual admitted runIDs. Shared A12 drawj for primary fresh-v01 and secondary A8a contrasts with frozen comparator drawj. Mean of run fractions; unadjusted sorted endpoints249/9749, no repeat-number or item pairing. Next canonical n5 block reserved for A13.',
    limits=['Five evaluations of one trained checkpoint; few run clusters and unadjusted intervals do not estimate across-training variability.', 'Four original infrastructure-incomplete workflows retained with seven unscoredNULL rows; only one authorized fresh whole replacement each admitted, no best-of, partial pooling or missing-as-zero.', 'A12 changes cumulative exposure/horizon versus A8a; no isolated causal identity/reasoning mechanism or release inference.', 'New production evaluations versus older interactive comparators are non-time-randomized; priority/latency is not a proven score cause.', 'Original slot1 diagnostic cohort remains the original incomplete workflow, never its admitted replacement.', 'Typed Arrow/JSON are full native-reader projections, not original sampleParquet bytes or independent remote verifier recomputation. Capability URLs stay restricted; global mirror fields do not attest peer execution policy.'],
    model_network_storage_control_calls=0, new_retry_course_or_release_authority=False,
    source_script_sha256=digest(Path(__file__).read_bytes()),
)
destination = HERE / 'A12-run-bootstrap-extension-marten-r1.json'
with destination.open('x') as stream:
    json.dump(output, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(path=str(destination), sha256=digest(destination.read_bytes()), summary=summary, contrasts=contrasts, A13_start=output['A13_canonical_block_start_state_sha256']), indent=2))
