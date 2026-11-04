import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import pyarrow.ipc as ipc


ROOT = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
HERE = ROOT / 'bfcl-noise-r1'
EXECUTION = ROOT / 'second-agentic-swe100-production-topup-execution-r1'
OLD_SHA = 'd7087cefa0beba30894b80aba95717d3d78ca08ad3128529a02f236c2a9b96e2'
PREREG_SHA = 'c221c5bb9f1c49f9cb2dd4152481aefd97b61c040d55740d8cc040ce756b6b9d'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected, length=None):
    raw = Path(path).read_bytes()
    assert digest(raw) == expected, path
    assert length is None or len(raw) == length, path
    return json.loads(raw)


def entry_checked(entry):
    return checked(entry['path'], entry['sha256'], entry['bytes'])


def interval(values):
    ordered = sorted(values)
    assert len(ordered) == 10000
    return [100 * ordered[249], 100 * ordered[9749]]


old_path = HERE / 'SWE100-reduced-n-bootstrap-extension-marten-r1.json'
old = checked(old_path, OLD_SHA)
prereg = checked(HERE / 'SWE100-topup-and-labelled99-view-preregistration-marten-r1.json', PREREG_SHA)
sibling_admission = checked(EXECUTION / 'A8a/topup-run-4/fixed100-full-native-grading-config-and-attempts-admission-margay-r1.json', 'e42f8bfe4e1757db3027a956b3e5ef8d8e410d45bc45b994610cac1dbecfde1a')
assert sibling_admission['logical_full100_admissible'] and sibling_admission['scored_binary_rewards'] == 100
assert old['summaries']['A8a']['n'] == 2
readback = checked(ROOT / 'second-agentic-swe100-execution-r1/final-seven-FULL100-native-record-and-full700-samples-readback-margay-r1.json', old['actual_native_readback_sha256'])
baseline_entry = next(row for row in readback['rows'] if row['arm'] == 'A3')
baseline_native = entry_checked(baseline_entry['native_record_full_original_restricted_local'])
registry = checked(ROOT / 'second-agentic-swe100-execution-r1/registered12-logical-slots-and-fixed100tasks-margay-r1.json', '52eb8469895cbd7a96e5f0aca6ad996f5e22c663f076cabcc64a09b06ace1a59')
tasks = set(registry['fixed100_task_names'])
assert len(tasks) == 100
runs = list(old['summaries']['A3']['runs'])
qualification = []
for slot, expected_sha in [(4, 'fe2273ffdf91862b3fe43a5c1708369360d275bf05d1c3953bd5dadc07bc6a43'), (5, '175a2f5a82b49c2c0c65a70f0e7f5a227fde216b6488fe6524a2ed8f9ce6792b')]:
    folder = EXECUTION / 'A3' / f'topup-run-{slot}'
    admission = checked(folder / 'fixed100-full-native-grading-config-and-attempts-admission-margay-r1.json', expected_sha)
    launch = checked(folder / 'launch.json', admission['launch_sha256'])
    assert launch['job'] == admission['job']
    assert launch['source_pin'] == '43bba4a16d175072b20fea27b8fb5524c377c79f'
    assert launch['application_submit_calls'] == 1
    assert admission['arm'] == 'A3' and admission['original_slot'] == slot and admission['replacement_number'] == 0
    native = entry_checked(admission['native_record_full_original_restricted_local'])
    assert native['status'] == 'succeeded' and native['run_id'] == admission['native_eval_id']
    assert native['model'] == baseline_native['model']
    assert native['eval'] == baseline_native['eval']
    assert native['results_path'] == admission['results_uri'] == launch['records_prefix'].rstrip('/') + '/' + native['run_id'] + '/results'
    samples = entry_checked(admission['samples_full_JSON_native_reader_projection'])
    arrow_entry = admission['samples_full_typed_native_reader_projection']
    raw = Path(arrow_entry['path']).read_bytes()
    assert digest(raw) == arrow_entry['sha256'] and len(raw) == arrow_entry['bytes']
    table = ipc.open_file(arrow_entry['path']).read_all()
    assert json.loads(json.dumps(table.to_pylist())) == samples
    assert table.num_rows == len(samples) == 100
    observed = {}
    for sample in samples:
        rewards = [value for name, value in sample['metrics'] if name == 'reward']
        assert len(rewards) == 1
        reward = rewards[0]
        assert type(reward) in (int, float) and math.isfinite(reward) and reward in (0, 1)
        grade = sample['grading']
        detail = json.loads(grade['detail'])
        assert detail['scored'] is True
        assert grade['score'] == detail['rewards']['reward'] == reward
        assert detail['error'] is None
        assert sample['doc_id'] not in observed
        observed[sample['doc_id']] = reward
    assert set(observed) == tasks
    assert observed == {row['doc_id']: row['reward'] for row in admission['rows']}
    assert all(row['score_present'] for row in admission['rows'])
    coverage = native['coverage']['swebench-verified']
    assert all(coverage[key] == 100 for key in ['n_benchmark', 'n_attempted', 'n_scored'])
    assert coverage['errors'] == {}
    correct = int(sum(observed.values()))
    assert correct == admission['observed_correct'] == native['metrics']['swebench-verified']['solved']
    assert native['metrics']['swebench-verified']['total'] == 100
    runs.append(dict(slot=slot, run_id=admission['job'], correct=correct, source_admission_sha256=expected_sha, native_record_sha256=admission['native_record_full_original_restricted_local']['sha256']))
    qualification.append(dict(slot=slot, admission_sha256=expected_sha, native_record_sha256=admission['native_record_full_original_restricted_local']['sha256'], samples_JSON_sha256=admission['samples_full_JSON_native_reader_projection']['sha256'], samples_IPC_sha256=arrow_entry['sha256'], full100_scored_binary_fixedID_grading_joins=True, full_model_and_eval_config_exact_old_A3=True, scored_errors=[], source_only_policy_caveat='Global mirrored child status is not independent peer-policy or priority attestation; no effective child retry-policy claim from projection defaults.'))

runs.sort(key=lambda row: row['run_id'])
fractions = {label: [row['correct'] / 100 for row in old['summaries'][label]['runs']] for label in ['v01', 'A0', 'A3', 'A8a']}
generator = random.Random(297)
old_draws = {label: [] for label in fractions}
for iteration in range(10000):
    for label, values in fractions.items():
        old_draws[label].append(statistics.fmean(generator.choice(values) for cluster in range(len(values))))
assert digest(json.dumps(generator.getstate()).encode()) == old['after_suite_state_sha256']
for label, values in fractions.items():
    assert 100 * statistics.fmean(values) == old['summaries'][label]['mean_percent']
    if len(values) >= 2:
        assert interval(old_draws[label]) == old['summaries'][label]['unadjusted_mean_CI95_percent']
for label, comparator in [('A0', 'v01'), ('A3', 'v01'), ('A8a', 'v01'), ('A3', 'A0'), ('A8a', 'A0'), ('A8a', 'A3')]:
    if min(len(fractions[label]), len(fractions[comparator])) >= 2:
        assert interval([left - right for left, right in zip(old_draws[label], old_draws[comparator], strict=True)]) == old['contrasts'][f'{label}_minus_{comparator}']['unadjusted_CI95_pp']
start_state = generator.getstate()
values = [row['correct'] / 100 for row in runs]
branches = []
for sibling_n in (3, 4):
    branch_generator = random.Random()
    branch_generator.setstate(start_state)
    candidate_draws = []
    for iteration in range(10000):
        candidate_draws.append(statistics.fmean(branch_generator.choice(values) for cluster in range(len(values))))
        for cluster in range(sibling_n):
            branch_generator.choice(range(sibling_n))
    branches.append(dict(A8a_final_n=sibling_n, A3_mean_CI95_percent=interval(candidate_draws), A3_minus_v01_CI95_pp=interval([candidate - reference for candidate, reference in zip(candidate_draws, old_draws['v01'], strict=True)]), A3_draws_sha256=digest(json.dumps(candidate_draws).encode()), after_extension_state_sha256=digest(json.dumps(branch_generator.getstate()).encode())))
invariant_fields = ['A3_mean_CI95_percent', 'A3_minus_v01_CI95_pp']
invariant = all(branches[0][key] == branches[1][key] for key in invariant_fields)
output = dict(status='A3_CLOSED_NATIVE_QUALIFIED_SCALARS_FINAL_ACROSS_BOTH_REMAINING_REGISTERED_STREAM_BRANCHES' if invariant else 'A3_CLOSED_NATIVE_QUALIFIED_DESCRIPTIVES_FINAL_CI_PENDING_A8a_FINAL_N',
    preregistration_sha256=PREREG_SHA, original_primary_sha256=OLD_SHA,
    native_qualification=qualification, runs=runs, n=3, planned=5, exhausted_no_FULL100=2, pending_A3=0,
    old_runs_correct=[28], new_topup_runs_correct=[29, 28],
    mean_percent=100 * statistics.fmean(values), sample_SD_pp=100 * statistics.stdev(values), min_max_percent=[100 * min(values), 100 * max(values)],
    old_mean_percent=28.0, topup_only_mean_percent=28.5,
    A3_minus_v01_delta_pp=100 * (statistics.fmean(values) - statistics.fmean(fractions['v01'])),
    A3_minus_A0_delta_pp=100 * (statistics.fmean(values) - statistics.fmean(fractions['A0'])),
    A3_minus_A0_CI95_pp=None, A3_minus_A0_CI_null_reason='A0_has_only_one_complete_evaluation_cluster',
    mean_CI95_percent=branches[0]['A3_mean_CI95_percent'] if invariant else None,
    A3_minus_v01_CI95_pp=branches[0]['A3_minus_v01_CI95_pp'] if invariant else None,
    stream_scope='Exact c221 interleaved actual-n A3 then A8a draws. The only remaining count possibilities are A8a n3 if its pending slot exhausts, or n4 if admitted. Dummy index choices reproduce PRNG consumption without using or inventing the pending score. Both branches enumerated, no branch chosen. A8a scores have no effect on A3 draw consumption apart from final population size.',
    remaining_stream_branches=branches, A3_scalar_endpoints_identical_for_all_remaining_n=True if invariant else False,
    final_joint_PRNG_state=None, final_joint_PRNG_state_reason='Actual A8a chain is pending; no branch state selected or committed.',
    A8a_primary_or_A8a_minus_A3=None,
    interpretation='Complete-score-selected evaluation clusters from one trained checkpoint. Two old A3 chains remain exhausted, not zeros. Mixed historical interactive and post-outcome production top-ups; no priority-causal, equivalence, no-loss, or release claim. n3 uncertainty is coarse. No new contrast, item bootstrap, retry, model or storage operation.',
    independent_script_sha256=digest(Path(__file__).read_bytes()))
assert digest(old_path.read_bytes()) == OLD_SHA
destination = HERE / 'SWE100-A3-topup-native-and-bootstrap-extension-marten-r1.json'
with destination.open('x') as stream:
    json.dump(output, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(path=str(destination), sha256=digest(destination.read_bytes()), **{key: output[key] for key in ['status', 'mean_percent', 'sample_SD_pp', 'mean_CI95_percent', 'A3_minus_v01_delta_pp', 'A3_minus_v01_CI95_pp', 'A3_scalar_endpoints_identical_for_all_remaining_n']}), indent=2))
