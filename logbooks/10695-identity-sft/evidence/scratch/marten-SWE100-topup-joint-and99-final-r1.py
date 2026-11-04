import ast
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


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    raw = Path(path).read_bytes()
    assert digest(raw) == expected, path
    return json.loads(raw)


a3_report_path = HERE / 'SWE100-A3-topup-native-and-bootstrap-extension-marten-r1.json'
a3_report = checked(a3_report_path, '7a807d7138629babdbf3de3c6734eb773ed8a795880f101da8831846f6f1d364')
prefix_script = Path('/mnt/storage/karan-scratch/marten-SWE100-A3-topup-extension-r1.py')
assert digest(prefix_script.read_bytes()) == a3_report['independent_script_sha256']
tree = ast.parse(prefix_script.read_text())
prefix = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'branches' for target in node.targets):
        break
    prefix.append(node)
namespace = {'__file__': str(prefix_script)}
exec(compile(ast.Module(body=prefix, type_ignores=[]), str(prefix_script), 'exec'), namespace)
old = namespace['old']
old_draws = namespace['old_draws']
start_state = namespace['start_state']
old_path = HERE / 'SWE100-reduced-n-bootstrap-extension-marten-r1.json'
old_sha = digest(old_path.read_bytes())
admitted = {label: list(summary['runs']) for label, summary in old['summaries'].items()}
admitted['A3'] = namespace['runs']
readback = namespace['readback']
baseline_entry = next(row for row in readback['rows'] if row['arm'] == 'A8a')
baseline_native = namespace['entry_checked'](baseline_entry['native_record_full_original_restricted_local'])
tasks = namespace['tasks']
new_qualification = list(namespace['qualification'])
topup_admissions = {}
for label in ['A3', 'A8a']:
    for slot in [4, 5]:
        path = EXECUTION / label / f'topup-run-{slot}' / 'fixed100-full-native-grading-config-and-attempts-admission-margay-r1.json'
        expected = {('A3', 4): 'fe2273ffdf91862b3fe43a5c1708369360d275bf05d1c3953bd5dadc07bc6a43', ('A3', 5): '175a2f5a82b49c2c0c65a70f0e7f5a227fde216b6488fe6524a2ed8f9ce6792b', ('A8a', 4): 'e42f8bfe4e1757db3027a956b3e5ef8d8e410d45bc45b994610cac1dbecfde1a', ('A8a', 5): 'f2d3dd2422ec355b4640988a353b40ebfcdce686ab6e4a21195c7ead2dddcd0f'}[label, slot]
        admission = checked(path, expected)
        topup_admissions[label, slot] = admission
        if label == 'A3':
            continue
        launch = checked(path.with_name('launch.json'), admission['launch_sha256'])
        assert launch['job'] == admission['job'] and launch['application_submit_calls'] == 1
        assert launch['source_pin'] == '43bba4a16d175072b20fea27b8fb5524c377c79f'
        native = namespace['entry_checked'](admission['native_record_full_original_restricted_local'])
        assert native['status'] == 'succeeded' and native['run_id'] == admission['native_eval_id']
        assert native['model'] == baseline_native['model'] and native['eval'] == baseline_native['eval']
        assert native['results_path'] == admission['results_uri'] == launch['records_prefix'] + '/' + native['run_id'] + '/results'
        samples = namespace['entry_checked'](admission['samples_full_JSON_native_reader_projection'])
        arrow = admission['samples_full_typed_native_reader_projection']
        raw = Path(arrow['path']).read_bytes()
        assert digest(raw) == arrow['sha256'] and len(raw) == arrow['bytes']
        table = ipc.open_file(arrow['path']).read_all()
        assert json.loads(json.dumps(table.to_pylist())) == samples
        assert table.num_rows == len(samples) == 100
        observed = {}
        for sample in samples:
            values = [value for name, value in sample['metrics'] if name == 'reward']
            assert len(values) == 1
            reward = values[0]
            assert type(reward) in (int, float) and math.isfinite(reward) and reward in (0, 1)
            grade = sample['grading']
            detail = json.loads(grade['detail'])
            assert detail['scored'] is True and detail['error'] is None
            assert grade['score'] == detail['rewards']['reward'] == reward
            assert sample['doc_id'] not in observed
            observed[sample['doc_id']] = reward
        assert set(observed) == tasks
        assert observed == {row['doc_id']: row['reward'] for row in admission['rows']}
        assert all(row['score_present'] for row in admission['rows'])
        coverage = native['coverage']['swebench-verified']
        assert all(coverage[key] == 100 for key in ['n_benchmark', 'n_attempted', 'n_scored']) and coverage['errors'] == {}
        correct = int(sum(observed.values()))
        assert correct == admission['observed_correct'] == native['metrics']['swebench-verified']['solved']
        assert native['metrics']['swebench-verified']['total'] == 100
        admitted[label].append(dict(slot=slot, run_id=admission['job'], correct=correct, source_admission_sha256=expected, native_record_sha256=admission['native_record_full_original_restricted_local']['sha256']))
        new_qualification.append(dict(arm=label, slot=slot, admission_sha256=expected, native_record_sha256=admission['native_record_full_original_restricted_local']['sha256'], full_JSON_sha256=admission['samples_full_JSON_native_reader_projection']['sha256'], typed_IPC_sha256=arrow['sha256'], full100_grading_fixedIDs_rewards_and_original_model_eval_config_joins=True, scored_errors=[]))
for rows in admitted.values():
    rows.sort(key=lambda row: row['run_id'])
assert {label: len(rows) for label, rows in admitted.items()} == {'v01': 3, 'A0': 1, 'A3': 3, 'A8a': 4}
fractions = {label: [row['correct'] / 100 for row in rows] for label, rows in admitted.items()}
generator = random.Random()
generator.setstate(start_state)
draws = {label: list(old_draws[label]) for label in ['v01', 'A0']}
draws.update(A3=[], A8a=[])
for iteration in range(10000):
    for label in ['A3', 'A8a']:
        values = fractions[label]
        draws[label].append(statistics.fmean(generator.choice(values) for cluster in range(len(values))))
branch = next(row for row in a3_report['remaining_stream_branches'] if row['A8a_final_n'] == 4)
assert digest(json.dumps(draws['A3']).encode()) == branch['A3_draws_sha256']
assert digest(json.dumps(generator.getstate()).encode()) == branch['after_extension_state_sha256']
interval = namespace['interval']
summaries = {}
for label, values in fractions.items():
    planned = 3 if label in ['v01', 'A0'] else 5
    summaries[label] = dict(n=len(values), planned=planned, exhausted_no_FULL100=planned - len(values), pending=0, runs=admitted[label], mean_percent=100 * statistics.fmean(values), sample_SD_pp=100 * statistics.stdev(values) if len(values) >= 2 else None, min_max_percent=[100 * min(values), 100 * max(values)], mean_CI95_percent=interval(draws[label]) if len(values) >= 2 else None)
contrasts = {'A0_minus_v01': old['contrasts']['A0_minus_v01']}
for label, comparator in [('A3', 'v01'), ('A8a', 'v01'), ('A3', 'A0'), ('A8a', 'A0'), ('A8a', 'A3')]:
    eligible = min(len(fractions[label]), len(fractions[comparator])) >= 2
    contrasts[f'{label}_minus_{comparator}'] = dict(delta_pp=100 * (statistics.fmean(fractions[label]) - statistics.fmean(fractions[comparator])), unadjusted_CI95_pp=interval([left - right for left, right in zip(draws[label], draws[comparator], strict=True)]) if eligible else None, CI_null_reason=None if eligible else 'fewer_than_two_complete_evaluation_clusters')
assert summaries['A3']['mean_CI95_percent'] == a3_report['mean_CI95_percent']
assert contrasts['A3_minus_v01']['unadjusted_CI95_pp'] == a3_report['A3_minus_v01_CI95_pp']
index_path = ROOT / 'second-agentic-swe100-execution-r1/logical-physical-run-index-20261030T210138.506528Z.json'
index = checked(index_path, 'fe5f19c1ba978ddb4ae77f6c5bae6c5c64656045c63863882cc3bce1ce948872')
physical = {row['job']: row for row in index['physical_workflows']}
selected = {label: [] for label in admitted}
unavailable = []
for logical in index['logical_slots']:
    label, slot = logical['arm'], logical['logical_slot']
    assert not logical['state'].startswith('PENDING')
    chain = [physical[job] for job in logical['all_physical_jobs_retained']]
    assert all(row['SDK_terminal'] for row in chain)
    valid = [row for row in chain if row.get('native_full100_admission') and row['native_full100_admission']['logical_full100_admissible']]
    assert len(valid) <= 1
    chosen = valid[0] if valid else chain[-1]
    admission = chosen.get('native_full100_admission')
    if not admission:
        unavailable.append(dict(arm=label, slot=slot, last_physical_job=chosen['job'], reason='Last_authorized_terminal_attempt_has_no_complete99_or100_fixedID_score_table'))
        continue
    checked(admission['source_admission_path'], admission['source_admission_sha256'])
    rows = checked(admission['array_path'], admission['array_sha256'])['rows'] if 'array_path' in admission else admission['rows']
    assert len(rows) == len({row['doc_id'] for row in rows}) == 100 and {row['doc_id'] for row in rows} == tasks
    scored = [row for row in rows if row['score_present']]
    missing = [row for row in rows if not row['score_present']]
    assert all(row['reward'] is None for row in missing)
    assert all(type(row['reward']) in (int, float) and row['reward'] in (0, 1) for row in scored)
    if not valid:
        assert len(scored) == 99 and len(missing) == 1
        assert admission['native_coverage']['errors']['VerifierRuntimeError'] == 1
    assert len(scored) in [99, 100]
    selected[label].append(dict(slot=slot, job=chosen['job'], selected_rule='First_sole_FULL100' if valid else 'Last_authorized_terminal_attempt_after_chain_closure', scored=len(scored), unknown=len(missing), correct=int(sum(row['reward'] for row in scored)), unknown_docIDs=[row['doc_id'] for row in missing], native_record_sha256=admission.get('native_record_sha256', admission.get('record_sha256')), source_admission_sha256=admission['source_admission_sha256']))
for (label, slot), admission in topup_admissions.items():
    selected[label].append(dict(slot=slot, job=admission['job'], selected_rule='First_sole_FULL100', scored=100, unknown=0, correct=admission['observed_correct'], unknown_docIDs=[], native_record_sha256=admission['native_record_full_original_restricted_local']['sha256']))
descriptive = {}
for label, records in selected.items():
    records.sort(key=lambda row: row['slot'])
    planned = summaries[label]['planned']
    lower_total = sum(row['correct'] / 100 for row in records)
    upper_total = sum((row['correct'] + row['unknown']) / 100 for row in records)
    descriptive[label] = dict(selected_logical_records=len(records), planned=planned, unavailable_whole_slots=planned - len(records), records=records, available_case_equal_logical_mean_percent=100 * statistics.fmean(row['correct'] / row['scored'] for row in records), selected_record_deterministic_bounds_percent=[100 * lower_total / len(records), 100 * upper_total / len(records)], all_planned_slot_deterministic_bounds_percent=[100 * lower_total / planned, 100 * (upper_total + planned - len(records)) / planned])
reference = descriptive['v01']
supplement_contrasts = {}
for label in ['A0', 'A3', 'A8a']:
    model = descriptive[label]
    supplement_contrasts[label + '_minus_v01'] = dict(available_case_delta_pp=model['available_case_equal_logical_mean_percent'] - reference['available_case_equal_logical_mean_percent'])
    for key in ['selected_record_deterministic_bounds_percent', 'all_planned_slot_deterministic_bounds_percent']:
        supplement_contrasts[label + '_minus_v01'][key.replace('_percent', '_pp')] = [model[key][0] - reference[key][1], model[key][1] - reference[key][0]]
assert len(unavailable) == 1 and unavailable[0]['arm'] == 'A3' and unavailable[0]['slot'] == 3
assert digest(old_path.read_bytes()) == old_sha
primary = dict(status='FINAL_EXTENDED_SWE100_n3_1_3_4_ALL_FOUR_NEW_FULL100_NATIVE_QUALIFIED', preregistration_sha256=namespace['PREREG_SHA'], original_primary_sha256=old_sha, A3_previously_final_scalar_extension_sha256=digest(a3_report_path.read_bytes()), original_index_sha256=digest(index_path.read_bytes()), original_physical=27, new_physical=4, total_physical=31, planned_logical=16, complete_FULL100=11, exhausted_without_FULL100=5, pending=0, new400_native_qualification=new_qualification, summaries=summaries, contrasts=contrasts, old_suite_end_state_sha256=old['after_suite_state_sha256'], after_topup_actual_n4_state_sha256=digest(json.dumps(generator.getstate()).encode()), actual_A8a_n4_branch_matches_frozen_A3_draw_and_end_state=True, stream='Exact c221: replay old suite-only Random297 stream, retain old v01/A0 drawj, continue10000 iterations A3 then A8a at actual n3/n4. Lexical actual scored runIDs, no repeat pairing or task resampling; sorted endpoints249/9749. n1-related CIs and SD remain NULL. Original d708 unchanged.', limits=['Original interactive and new production cohorts combined after outcome-informed top-up choice; no priority-causal comparison.', 'One checkpoint per arm; complete-run selection need not be missing-at-random. Planned and exhausted chains retained.', 'Original700 native qualification carried by d708; all400 new native grading/records/typedJSON/config joins independently replayed here.', 'Global mirrored child settings are not authoritative effective peer-policy attestation; no inherited mirror-policy PASS.', 'No equivalence, noninferiority, no-loss, release, new comparison or further retry claim.'], script_sha256=digest(Path(__file__).read_bytes()))
supplement = dict(status='LABELLED_99_100_DESCRIPTIVE_AVAILABLE_CASE_AND_DETERMINISTIC_BOUNDS_ONLY_NOT_PRIMARY', preregistration_sha256=namespace['PREREG_SHA'], original_index_sha256=digest(index_path.read_bytes()), selection='Exactly one record per logicalslot; soleFULL100 dominates all earlier partials. Otherwise select last authorized terminal physical only if exact99 scored+1NULL. No earlier higher-score or higher-coverage choice.', model_views=descriptive, contrasts_only_vs_v01=supplement_contrasts, unavailable=unavailable, empirical_uncertainty_CI_or_test=None, source_scope='Four original terminal99 native reward/coverage/admission projections and all fixed100 docIDs checked locally; unscoredVerifierRuntime attribution comes from hash-bound native admissions, not a newly repeated full-grading or remote read. FULL100 rows use parent-qualified d708 plus new400 native grading checks. Native-scored agent timeouts retain actualzero; NULL never imputed.', limits='Bounds are deterministic extrema, not CIs. Available-case s/k mixes denominators99/100; planned-slot bounds additionally assign each absent entire record [0,1]. This view was requested after historical99 outcomes were known. No primary promotion, acceptance, best-of, score repair or new evaluation.', script_sha256=digest(Path(__file__).read_bytes()))
for name, output in [('SWE100-production-topup-joint-bootstrap-extension-marten-r1.json', primary), ('SWE100-labelled99-available-case-and-bounds-marten-r1.json', supplement)]:
    destination = HERE / name
    with destination.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(path=str(destination), sha256=digest(destination.read_bytes()))))
print(json.dumps(dict(primary_means={label: summary['mean_percent'] for label, summary in summaries.items()}, primary_contrasts=contrasts, descriptive99={label: {key: value for key, value in view.items() if key != 'records'} for label, view in descriptive.items()}), indent=2))
