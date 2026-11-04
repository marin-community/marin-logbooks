import base64
import copy
import hashlib
import json
import re
from pathlib import Path


BASE = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
PARENT = Path('/mnt/storage/karan-scratch/marten-A3-paired-maths-pending-r1.py')
TARGETS = {
    'M51': {
        'index': 'b454f4e93a7f2bb8952cc1d12f0020f51989ab59ca76c54e7959d1b554ee3637',
        'runtime': '6ec70558cadafaf5cf664d9d7ea3b1f087c60e64f5763bcca328afad71ca2b27',
        'binding': '7d719c236cfaaad9ee0278e96c1be2d75b55f3147d9188888ef6c361956e31ad',
        'science': '108c2f65840ea7ab22acdb677494726a35d316e7400af5ba048337e84d44c4b9',
    },
    'A12': {
        'index': '5bd49b6af043d2b8d4e1638c7cdb070eccf69585200d02232736e29b15173054',
        'runtime': '11f4496916fd4aac260774b8163a74e052e23296af0fac535b2aa8837a1a29a3',
        'binding': '04fe1177b93b5117bb971bc17aa4560d1381382b723dea6593d6df9c22f5a57f',
        'science': 'd3c966d90391dd7e1fabb5e6e29ecdb3382f454c71fd3a4a312878d03cea8738',
    },
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected, length=None):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, str(path)
    assert length is None or len(raw) == length, str(path)
    return raw


def descriptor(entry):
    return checked(entry['path'], entry['sha256'], entry['bytes'])


def runtime_checks(label, settings):
    root = BASE / (label + '-paired-math-r1')
    binding = json.loads(checked(root / 'binding-review-marten-r1.json', settings['binding']))
    plan = json.loads(checked(root / 'launch-plan.json', binding['plan_sha256']))
    assert digest(plan['preregistration']['path']) == settings['science']
    target = plan['targets'][label]
    assert digest(root / (label + '-candidate.json')) == target['candidate_sha256']
    qualification = json.loads(checked(root / 'actual-runtime-source-server-and-sampling-qualified-margay-r1.json', settings['runtime']))
    evidence = json.loads(checked(qualification['runtime_evidence_path'], qualification['runtime_evidence_sha256']))
    index = json.loads(checked(root / 'six-candidate-native-config-coverage-item-arrays-and-cost-margay-r1.json', settings['index']))
    assert evidence['candidate_native6_index_sha256'] == qualification['candidate6_native_index_sha256'] == settings['index']
    model = target['candidate']['model_path']
    assert qualification['actual_HF'] == index['actual_model_HF'] == model
    jobs = {(unit['trial'], unit['task']): unit['actual_eval_job'] for unit in index['units']}
    assert len(jobs) == 6 and len(evidence['sampling_markers']) == 6
    assert {(marker['trial'], marker['task']) for marker in evidence['sampling_markers']} == set(jobs)
    for marker in evidence['sampling_markers']:
        assert marker['job'] == jobs[(marker['trial'], marker['task'])]
        raw = json.loads(descriptor(marker['original_marker_bytes_restricted_local']))
        assert raw['job'] == marker['job'] and raw['lines'] == marker['first3_actual_marker_lines']
        assert len(raw['lines']) == 3
        expected = {'max_tokens': 8192, 'temperature': 0}
        if marker['trial'] != 'greedy':
            expected = {'max_tokens': 8192, 'temperature': 0.7, 'top_p': 0.95, 'seed': int(marker['trial'])}
        for line in raw['lines']:
            assert json.loads(line.split('TASK297_FINAL_SAMPLING_PAYLOAD ', 1)[1]) == expected
    records = {record['job']: record for record in evidence['actual_job_RPC_and_log_evidence']}
    assert len(qualification['six_actual_clients']) == 6
    for unit in qualification['six_actual_clients']:
        assert unit['job'] == jobs[(unit['trial'], unit['task'])]
        record = records[unit['job']]
        assert record['state'] == 'succeeded'
        lines = record['snippets']['model_args']
        line = next(line for line in lines if 'running evalchemy:' in line)
        assert 'model=' + model + ',base_url=[REDACTED_URL],' in line
        for fragment in ['tokenizer_backend=none,', 'tokenized_requests=False,', 'num_concurrent=8,', 'max_length=32704', 'max_gen_toks=8192']:
            assert fragment in line
        encoded = re.search(r'chat_template_kwargs=base64:([^, ]+)', line).group(1)
        assert json.loads(base64.b64decode(encoded)) == {'enable_thinking': unit['task'] == 'math500'}
        assert any('Using tokenizer None' in line for line in record['snippets']['tokenizer'])
        assert any('served max_model_len: 32768 (lm-eval max_length=32704)' in line for line in record['snippets']['max_length'])
    serve = next(record for record in records.values() if record['job'].endswith('-serve'))
    text = json.dumps(serve['snippets'])
    for fragment in ["'model_tag': '" + model, "'model': '" + model, "'tokenizer': '" + model, "'dtype': 'bfloat16'", "'max_model_len': 32768", "'reasoning_parser': 'grug'", "'data_parallel_size': 8", "'enable_expert_parallel': True", 'tensor_parallel_size=1', '01911be34fac']:
        assert fragment in text, fragment
    members = {}
    for archive in evidence['owner_archives']:
        for member in archive['members']:
            raw = descriptor(member['original_restricted_local'])
            assert descriptor(member['sanitized_projection']) == raw
            members[member['member']] = json.loads(raw)
    assert len(members) == 8
    assert members['candidate.json']['candidate'] == target['candidate']
    assert members['candidate.json']['sha256'] == target['candidate_sha256']
    assert members['source-bundles.json']['math_archive_sha256'] == plan['source_archive']['sha256']
    launch = json.loads(checked(root / (label + '-actual-launch-margay-r1.json'), qualification['actual_launch_sha256']))
    assert launch['code_pin'] == plan['code_pin'] == binding['source_pin']
    checked(plan['actual_readiness']['path'], plan['actual_readiness']['sha256'])
    assert plan['actual_readiness']['sha256'] == binding['actual_readiness_sha256']
    return qualification


assert digest(PARENT) == '06d9c4d18ce3a5cf179b7fa2bb2f8b9c574cd2faff8f65586a4976bdb8f8d5bd'
for label, settings in TARGETS.items():
    qualification = runtime_checks(label, settings)
    source = PARENT.read_text().replace("ROOT = BASE / 'A3-paired-math-infra-replacement1-r1'", "ROOT = BASE / '" + label + "-paired-math-r1'")
    source = source.replace('A3', label)
    source = source.replace(label + '-A11-paired-maths-preregistration-marten-r1.json', 'A3-A11-paired-maths-preregistration-marten-r1.json')
    source = source.replace('824aa6b52dbb36c947b7cd6c0728ac8e06bd2ba42a2db2683a738089b347ee7f', settings['index'])
    source = source.replace('six-candidate-native-config-coverage-item-arrays-margay-r1.json', 'six-candidate-native-config-coverage-item-arrays-and-cost-margay-r1.json')
    source = source.replace('First zero-output preempted ' + label + ' physical attempt remains separate, no preferred-success selection', 'One candidate-only maths workflow; trained/export or merge histories remain separate, no new reference or preferred-success selection')
    exec(compile(source, str(PARENT), 'exec'), {'__file__': __file__, '__name__': '__main__'})
    root = BASE / (label + '-paired-math-r1')
    conditional_path = root / 'paired-math-statistics-pending-runtime-marten-r1.json'
    conditional = json.loads(conditional_path.read_text())
    results = copy.deepcopy(conditional['results'])
    for result in results.values():
        contrast = result[label + '_minus_existing_fresh_v01']
        result['registered_stop_triggered'] = contrast['delta_pp'] < -1.5 and contrast['ci95_pp'][1] < 0
        result.pop('stop_or_quality_decision')
        result.pop('reason')
    output = dict(
        status='FINAL_REGISTERED_PAIRED_MATHS_RUNTIME_QUALIFIED', label=label,
        science_preregistration_sha256=settings['science'], parent_method_sha256=conditional['parent_method_sha256'],
        candidate_extension_preregistration_sha256=conditional['preregistration_sha256'],
        conditional_statistics_sha256=digest(conditional_path), actual_runtime_qualification_sha256=settings['runtime'],
        runtime_evidence_sha256=qualification['runtime_evidence_sha256'], binding_review_sha256=settings['binding'],
        reference_analysis_sha256='c410fda8a75cf1ab2b9e2231beec8060062f9d462232e861d53cd8d45925b2df',
        reused_reference_index_sha256=conditional['reused_reference_index_sha256'],
        runtime_independent_checks='All18 original marker byte/dictionary joins; six actual client modes/HF/backend/concurrency/gen/context/tokenizer logs; actual server HF=tokenizer/BF16/TP1DP8EP/grug32768/01911; eight owner member original/safe byte equalities; committed wrapped candidate/source archive/launch/source pin/own readiness joins. All12 native done/config/metric/coverage/item arrays and two paired NPZs checked by hash-bound parent.',
        results=results, statistics=conditional['statistics'],
        any_registered_stop_triggered=any(result['registered_stop_triggered'] for result in results.values()),
        interpretation='Only the preregistered delta_pp<-1.5 AND upper95CI_pp<0 triggers STOP. Absence of STOP does not establish noninferiority, no loss or release acceptance. No candidate-candidate contrast added.',
        limits=['One checkpoint per condition/fixed public items/three decodes; paired item resamples are not replicated training runs.', 'Existing admitted fresh-v01 six units reused, no new reference serve or historical pooling.', 'First18 marker lines are not full wire HTTP or all5457 candidate payloads.', 'Server32768/client32704 retains frozen64-token margin; tokenizer_backend none/tokenized_requests False.', 'Global federated child RPC mirror fields do not attest original peer effective policy/priority/workdir; no runtime childZIP equality inferred.', 'Item arrays are native-reader projections checked against native done/headline/config/coverage, not independent remote per-item verifier rereads.'],
        old_analysis_mutation=False, model_network_storage_control_calls=0,
        finalization_script_sha256=digest(__file__),
    )
    destination = root / 'paired-math-analysis-marten-r1.json'
    with destination.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(label=label, path=str(destination), sha256=digest(destination), results=results), indent=2))
