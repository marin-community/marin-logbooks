import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


ARMS = ('A0', 'A1', 'A2', 'A3', 'A4')
EXPECTED = {'A0': 6, 'A1': 5, 'A2': 5, 'A3': 5, 'A4': 5}
ROOT = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
SPEC = ROOT / 'bfcl-noise-r1/run-mean-analysis-spec-marten-r1.json'
SPEC_SHA256 = '636ddeb0a79606fc4827c1138668d8383dcd1fec387e5bbc8cd0d898f2080b45'
PRIOR_INDEX = ROOT / 'bfcl-repeat20-r1/all34-physical-and26-logical-run-index-margay-r3.json'
PRIOR_INDEX_SHA256 = '24ae96d1487a92dacb8c80773ec298205731cbadefcbf409985972c0c5121bd0'
GO_LAUNCHES = ROOT / 'bfcl-missing7-go-r1/all-seven-attempt1-launch-margay-r1.json'
GO_LAUNCHES_SHA256 = '3a4e920f2448ce5bbdcc673b99a91e5013e290c7696c60360a653653c2f2d632'


def checked_json(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f'Changed input: {path}')
    return json.loads(raw)


def interval(values):
    ordered = sorted(values)
    return [ordered[249], ordered[9749]]


def bootstrap(arms):
    if {arm: len(arms[arm]) for arm in ARMS} != EXPECTED:
        raise ValueError('Complete6/5/5/5/5 runs required before primary bootstrap')
    fractions = {arm: [count / 123 for count in arms[arm]] for arm in ARMS}
    generator = random.Random(297)
    samples = {arm: [] for arm in ARMS}
    contrasts = {arm: [] for arm in ARMS[1:]}
    for draw in range(10000):
        means = {}
        for arm in ARMS:
            means[arm] = statistics.fmean(generator.choice(fractions[arm]) for cluster in range(EXPECTED[arm]))
            samples[arm].append(means[arm])
        for arm in ARMS[1:]:
            contrasts[arm].append(means[arm] - means['A0'])
    summaries = {}
    for arm in ARMS:
        values = fractions[arm]
        summaries[arm] = dict(n=len(values), success_counts=arms[arm], mean_fraction=statistics.fmean(values),
            mean_percent=100 * statistics.fmean(values), sample_SD_percent=100 * statistics.stdev(values),
            min_max_percent=[100 * min(values), 100 * max(values)],
            unadjusted_mean_CI95_percent=[100 * value for value in interval(samples[arm])])
    comparison = {arm + '_minus_A0': dict(
        delta_pp=100 * (statistics.fmean(fractions[arm]) - statistics.fmean(fractions['A0'])),
        unadjusted_CI95_pp=[100 * value for value in interval(contrasts[arm])]) for arm in ARMS[1:]}
    return dict(per_arm=summaries, contrasts=comparison, draws=10000, seed=297,
        sampling='Independent complete-run resamples within each arm, shared A0 draw across contrasts',
        CI_order_statistic_indices=[249, 9749], multiplicity_adjustment='none')


def physical_score_entry(row, physical_runs):
    physical = {entry['run_id']: entry for entry in physical_runs}
    if row['kind'] != 'LATEST_INTERACTIVE_GO_ATTEMPT1':
        if row != physical[row['run_id']]:
            raise ValueError('Logical score entry differs from its physical run')
        return row
    chain = row['full_physical_chain_retained']
    expected = [entry['run_id'] for entry in physical_runs if entry['logical_slot'] == row['logical_slot']]
    if chain != expected or len(chain) != 3 or chain[-1] != row['run_id']:
        raise ValueError('GO logical metadata does not preserve the exact three-attempt physical chain')
    score_entry = {key: value for key, value in row.items() if key != 'full_physical_chain_retained'}
    if score_entry != physical[row['run_id']]:
        raise ValueError('GO score entry differs from its physical run beyond reviewed chain metadata')
    return score_entry


def audit_expanded_physical_chain(index):
    prior = checked_json(PRIOR_INDEX, PRIOR_INDEX_SHA256)
    launches = checked_json(GO_LAUNCHES, GO_LAUNCHES_SHA256)
    physical = {row['run_id']: row for row in index['physical_runs']}
    primary = {row['logical_slot']: row for row in index['primary_runs']}
    old_physical = {row['run_id']: row for row in prior['physical_runs']}
    old_primary = {row['logical_slot']: row for row in prior['primary_runs']}
    if len(physical) != 41 or len(primary) != 26:
        raise ValueError('This version admits the exact34 prior physical runs plus7 authorized GOattempt1 runs')
    if any(physical.get(run_id) != row for run_id, row in old_physical.items()):
        raise ValueError('A previously scored run or infrastructure failure was mutated or removed')
    if set(physical) != set(old_physical) | {row['job'] for row in launches['rows']}:
        raise ValueError('Physical runs differ from the authorized launch set')
    missing_slots = {row['logical_slot'] for row in prior['primary_runs'] if row['state'] == 'INFRA_FAILED_NO_SCORE'}
    if missing_slots != {row['slot'] for row in launches['rows']} or len(missing_slots) != 7:
        raise ValueError('Authorized GO slots do not equal the seven missing slots')
    for slot, row in old_primary.items():
        if slot not in missing_slots and primary.get(slot) != row:
            raise ValueError('A valid scored logical slot was replaced')
    for linked in launches['rows']:
        row = physical[linked['job']]
        old = old_primary[linked['slot']]
        launch = checked_json(linked['launch'], linked['sha256'])
        if physical_score_entry(primary[linked['slot']], index['physical_runs']) != row or row['logical_slot'] != linked['slot']:
            raise ValueError('GO attempt is not the unique current run for its logical slot')
        if row['kind'] != 'LATEST_INTERACTIVE_GO_ATTEMPT1':
            raise ValueError('Unexpected GO attempt schema')
        if old['run_id'] != linked['prior_failed_job'] or launch['prior_failed_job'] != old['run_id']:
            raise ValueError('GO attempt changed its failed predecessor')
        if row['launch_sha256'] != linked['sha256'] or launch['job'] != row['run_id'] or launch['go_attempt'] != 1:
            raise ValueError('GO physical run differs from its exact launch')
        for field in ('HF', 'model_config_sha256', 'method_pin'):
            if row[field] != old[field]:
                raise ValueError('GO model or evaluator changed')
    failures = []
    for row in index['physical_runs']:
        if row['state'] != 'INFRA_FAILED_NO_SCORE':
            continue
        record = checked_json(row['failed_record'], row['failed_record_sha256'])
        if record['status'] != 'infra_failed' or record['metrics'] != {} or row['correct'] is not None:
            raise ValueError('Infrastructure failure assigned a score or metrics')
        failures.append(row['run_id'])
    return failures


def audit(index):
    primary = index['primary_runs']
    physical = index['physical_runs']
    if len(primary) != 26 or len(physical) != 41:
        raise ValueError('Expected26 logical slots and41 retained physical workflows')
    if len({row['logical_slot'] for row in primary}) != 26 or len({row['run_id'] for row in physical}) != 41:
        raise ValueError('Duplicate logical slot or physical run ID')
    if Counter(row['arm'] for row in primary) != EXPECTED:
        raise ValueError('Wrong per-arm planned counts')
    physical_by_id = {row['run_id']: row for row in physical}
    originals = {row['arm']: row for row in primary if row['kind'] == 'ORIGINAL'}
    if set(originals) != set(ARMS):
        raise ValueError('Missing registered original arm run')
    baseline_array = checked_json(originals['A0']['paired123'], originals['A0']['paired123_sha256'])
    reference = {row['doc_id']: row['reference_reward'] for row in baseline_array['rows']}
    if len(reference) != 123 or sum(reference.values()) != 34:
        raise ValueError('Reference coverage/count mismatch')
    scored = {arm: [] for arm in ARMS}
    pending = []
    admitted = []
    for logical_row in primary:
        row = physical_score_entry(logical_row, physical)
        if physical_by_id[row['run_id']] != row:
            raise ValueError('Primary run differs from retained physical run entry')
        arm = row['arm']
        for field in ('HF', 'model_config_sha256', 'method_pin'):
            if row[field] != originals[arm][field]:
                raise ValueError('Arm HF/model/method changed')
        if row['method_pin'] != 'caba0fa38b780af42da54af2f188ba5bfcc28501':
            raise ValueError('Evaluator pin changed')
        checked_json(row['launch'], row['launch_sha256'])
        model_raw = Path(row['model_config']).read_bytes()
        if hashlib.sha256(model_raw).hexdigest() != row['model_config_sha256']:
            raise ValueError('Model YAML changed')
        if row['kind'] != 'LATEST_INTERACTIVE_GO_ATTEMPT1' and row['replaces_failed_job']:
            failed = physical_by_id[row['replaces_failed_job']]
            if failed['logical_slot'] != row['logical_slot'] or failed['state'] != 'INFRA_FAILED_NO_SCORE':
                raise ValueError('Invalid startup replacement slot')
            record = checked_json(failed['failed_record'], failed['failed_record_sha256'])
            if record['status'] != 'infra_failed' or record['metrics'] != {}:
                raise ValueError('Replacement original unexpectedly scored')
        if row['state'] != 'COMPLETE123':
            if row['correct'] is not None:
                raise ValueError('Incomplete run given a numerical score')
            pending.append(dict(arm=arm, logical_slot=row['logical_slot'], run_id=row['run_id'], state=row['state']))
            continue
        array = checked_json(row['paired123'], row['paired123_sha256'])
        report = checked_json(row['report'], row['report_sha256'])
        if array['score_sha256'] != row['report_sha256'] or array['candidate_results_uri'] != row['results_uri']:
            raise ValueError('Array/report/native result URI mismatch')
        entries = array['rows']
        rewards = {entry['doc_id']: entry['candidate_reward'] for entry in entries}
        refs = {entry['doc_id']: entry['reference_reward'] for entry in entries}
        if len(entries) != 123 or len(rewards) != 123 or refs != reference:
            raise ValueError('Missing/duplicate/misaligned per-task coverage')
        if any(type(value) not in (float, int) or not math.isfinite(value) or value not in (0, 1) for value in rewards.values()):
            raise ValueError('Invalid reward')
        correct = int(sum(rewards.values()))
        if len(report) != 1:
            raise ValueError('Expected one native result report entry')
        result = next(iter(report.values()))
        reported = result.get('correct', result.get('classes', {}).get('pass'))
        if correct != row['correct'] or correct != reported or result['n'] != 123:
            raise ValueError('Per-run count/report mismatch')
        if not math.isclose(result['mean_pct'], correct / 123 * 100, rel_tol=0, abs_tol=1e-10):
            raise ValueError('Per-run mean mismatch')
        scored[arm].append((row['run_id'], correct))
        admitted.append(dict(arm=arm, logical_slot=row['logical_slot'], run_id=row['run_id'], correct=correct,
            array_sha256=row['paired123_sha256'], report_sha256=row['report_sha256']))
    for arm in ARMS:
        scored[arm].sort()
        if index['counts'][arm]['complete'] != len(scored[arm]):
            raise ValueError('Metadata complete count mismatch')
    return scored, pending, admitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--index-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specification = checked_json(SPEC, SPEC_SHA256)
    index = checked_json(args.index, args.index_sha256)
    failures = audit_expanded_physical_chain(index)
    scored, pending, admitted = audit(index)
    complete = not pending and {arm: len(scored[arm]) for arm in ARMS} == EXPECTED
    primary_analysis = bootstrap({arm: [count for run_id, count in scored[arm]] for arm in ARMS}) if complete else None
    report = dict(status='COMPLETE_PRIMARY_RUN_BOOTSTRAP' if complete else 'INCOMPLETE_NO_PRIMARY_MEANS_OR_CONTRASTS',
        index_path=str(args.index), index_sha256=args.index_sha256, specification_sha256=SPEC_SHA256,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        planned=EXPECTED, complete_counts={arm: len(scored[arm]) for arm in ARMS},
        physical_workflows=41, preserved_infrastructure_non_scores=len(failures),
        preserved_infrastructure_failure_run_ids=failures,
        authorized_GO_launch_sha256=GO_LAUNCHES_SHA256,
        prior34_index_sha256=PRIOR_INDEX_SHA256, admitted_runs=admitted, pending=pending,
        primary_analysis=primary_analysis, limitations=specification['limitations'],
        descriptive_reference_only='Historical34/123 has no replicated reference variance estimate',
        original_single_run_contrasts='Preserved, superseded for decisions',
        old_r5_diagnostic_excluded=True, retry_authority=False, model_or_remote_calls=0)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'status': report['status'], 'complete_counts': report['complete_counts']}, indent=2))


if __name__ == '__main__':
    main()
