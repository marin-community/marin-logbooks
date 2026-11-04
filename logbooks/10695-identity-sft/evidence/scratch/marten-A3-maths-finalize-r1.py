import base64
import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b/A3-paired-math-infra-replacement1-r1')
RUNTIME = ROOT / 'actual-runtime-qualification-r1'
HF = 's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-6ac2c53d0611/A3/run/hf/updates-36'


def checked(path, expected, length=None):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, path
    assert length is None or len(data) == length, path
    return data


def read_checked(name, expected):
    return json.loads(checked(RUNTIME / name, expected))


stats_sha = 'e50e8cd6bd264d4e734079ade358357c27ac9bf1decba456ac114feb76268541'
stats = json.loads(checked(ROOT / 'paired-math-statistics-pending-runtime-marten-r1.json', stats_sha))
evidence_sha = '95a3532e9478b7db599505235b68a686ff4db47c541ea0f99e9773a01290a62e'
evidence = read_checked('actual-owner-server-sixunits-first18-sampling-evidence-margay-r1.json', evidence_sha)
amendment_sha = '76943002f15b31f59d1bf09fc9429815fe1f23c72b92d9d532035268924ace69'
amendment = read_checked('all6-client-model-args-URL-scope-correction-margay-r1.json', amendment_sha)
qualification_sha = 'e6023fb2e521f5a7d5988312c6c838a97824ab39cb1921287822fc418c1c7a97'
qualification = read_checked('actual-runtime-qualification-margay-r1.json', qualification_sha)
assert qualification['source_evidence']['sha256'] == evidence_sha
assert qualification['client_scope_amendment']['sha256'] == amendment_sha
assert evidence['candidate_native6_index_sha256'] == stats['candidate_units_index_sha256']
units = json.loads(checked(ROOT / 'six-candidate-native-config-coverage-item-arrays-margay-r1.json', stats['candidate_units_index_sha256']))['units']
jobs = {(unit['trial'], unit['task']): unit['actual_eval_job'] for unit in units}
assert len(jobs) == 6
for coverage in stats['coverage']:
    checked(coverage['native_done_path'], coverage['native_done_sha256'])
    item = coverage['item_scores']
    checked(item['path'], item['sha256'], item['bytes'])
for result in stats['results'].values():
    arrays = result['aligned_arrays']
    checked(arrays['path'], arrays['sha256'], arrays['bytes'])

assert len(evidence['sampling_markers']) == 6
for marker in evidence['sampling_markers']:
    assert jobs[(marker['trial'], marker['task'])] == marker['job']
    original = marker['original_marker_bytes_restricted_local']
    raw = json.loads(checked(original['path'], original['sha256'], original['bytes']))
    assert raw['job'] == marker['job']
    assert raw['lines'] == marker['first3_actual_marker_lines']
    assert len(raw['lines']) == 3
    expected = {'max_tokens': 8192, 'temperature': 0}
    if marker['trial'] != 'greedy':
        expected = {'max_tokens': 8192, 'temperature': 0.7, 'top_p': 0.95, 'seed': int(marker['trial'])}
    for line in raw['lines']:
        assert json.loads(line.split('TASK297_FINAL_SAMPLING_PAYLOAD ', 1)[1]) == expected

assert len(amendment['rows']) == 6
for row in amendment['rows']:
    assert row['job'] == jobs[(row['trial'], row['task'])]
    line = row['URL_redacted_actual_line']
    assert f'model={HF},base_url=[REDACTED_URL],' in line
    for fragment in ('tokenizer_backend=none,', 'tokenized_requests=False,', 'num_concurrent=8,', 'max_length=32704'):
        assert fragment in line
    encoded = re.search(r'chat_template_kwargs=base64:([^, ]+)', line).group(1)
    assert json.loads(base64.b64decode(encoded)) == {'enable_thinking': row['task'] == 'math500'}

job_records = {item['job']: item for item in evidence['actual_job_RPC_and_log_evidence']}
for job in jobs.values():
    record = job_records[job]
    assert record['state'] == 'succeeded'
    assert any('Using tokenizer None' in line for line in record['snippets']['tokenizer'])
    assert any('served max_model_len: 32768 (lm-eval max_length=32704)' in line for line in record['snippets']['max_length'])
serve = next(item for item in job_records.values() if item['job'].endswith('-serve'))
serve_text = json.dumps(serve['snippets'])
for fragment in ("'model_tag': '" + HF, "'model': '" + HF, "'tokenizer': '" + HF, "'dtype': 'bfloat16'", "'max_model_len': 32768", "'reasoning_parser': 'grug'", "'data_parallel_size': 8", "'enable_expert_parallel': True", 'tensor_parallel_size=1'):
    assert fragment in serve_text, fragment
members = {}
for archive in evidence['owner_archives']:
    for member in archive['members']:
        source = member['original_restricted_local']
        safe = member['sanitized_projection']
        source_data = checked(source['path'], source['sha256'], source['bytes'])
        assert checked(safe['path'], safe['sha256'], safe['bytes']) == source_data
        members[member['member']] = json.loads(source_data)
candidate = members['candidate.json']['candidate']
assert candidate['model_path'] == HF
assert members['source-bundles.json']['math_archive_sha256'] == '2f0d5ce76bb408f0e803bd51044472af11ddb77b364443d9b09c1a5f5ab7379d'

results = copy.deepcopy(stats['results'])
for result in results.values():
    contrast = result['A3_minus_existing_fresh_v01']
    result['registered_stop_triggered'] = contrast['delta_pp'] < -1.5 and contrast['ci95_pp'][1] < 0
    result.pop('stop_or_quality_decision')
    result.pop('reason')
assert results['gsm8k_0shot']['registered_stop_triggered']
assert not results['math500']['registered_stop_triggered']
output = {
    'status': 'FINAL_REGISTERED_PAIRED_MATHS_A3_GSM_STOP_TRIGGERED',
    'preregistration_sha256': stats['preregistration_sha256'],
    'conditional_statistics_sha256': stats_sha,
    'actual_runtime_qualification_sha256': qualification_sha,
    'runtime_evidence_sha256': evidence_sha,
    'client_scope_amendment_sha256': amendment_sha,
    'reused_reference_index_sha256': stats['reused_reference_index_sha256'],
    'runtime_independent_checks': 'Six actual unit/job joins; all18 marker dictionaries and raw file hashes; all six CLI mode/backend/context controls; actual server model/tokenizer/HF/dtype/parallelism/context; eight owner member bytes equal safe projections; all12 native done/item-array hashes and two aligned NPZ hashes',
    'results': results,
    'statistics': stats['statistics'],
    'any_registered_stop_triggered': True,
    'stop_interpretation': 'Registered rule uses point delta below -1.5pp AND upper95CI below zero. GSM meets both. This interval does not establish true loss exceeds1.5pp. No automatic cancellation, release acceptance, or new experiment authority.',
    'limits': [
        'One checkpoint per condition, fixed public items and three decodes; uncertainty resamples paired item means, not independently trained models.',
        'Existing admitted fresh-v01 six units reused; original zero-output preempted candidate attempt preserved separately.',
        'First18 finalized sampling markers are not full wire HTTP or all5457 candidate payloads.',
        'Actual server context32768 and client32704 differ by the pinned64-token margin; client tokenizer_backend none and tokenized_requests False.',
        'Stripped child RPC does not establish live worker member identity or effective retry policy. No runtime child ZIP equality inferred.',
        'Item arrays are hash-bound local native-reader projections checked against native done/coverage/headline metrics, not an independent remote per-item verifier reread.'
    ],
    'no_new_reference_serve': True,
    'no_old_analysis_mutation': True,
    'model_network_storage_control_calls': 0,
    'finalization_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
destination = ROOT / 'paired-math-analysis-marten-r1.json'
with destination.open('x') as stream:
    json.dump(output, stream, indent=2)
    stream.write('\n')
print(json.dumps({'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'GSM_registered_stop': True, 'MATH_registered_stop': False}))
