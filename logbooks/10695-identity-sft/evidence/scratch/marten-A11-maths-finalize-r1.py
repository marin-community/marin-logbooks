import base64
import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b/A11-paired-math-r1')
HF = 's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/bfcl-ablation-741e5928030c/A11/run/hf/updates-72'


def checked(path, expected, length=None):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, path
    assert length is None or len(raw) == length, path
    return raw


stats_sha = '3f1860e4be062f2f6363e0a93adabc0a04cba64aa01a127b7f27fd11735dd20f'
stats = json.loads(checked(ROOT/'paired-math-statistics-pending-runtime-marten-r1.json',stats_sha))
qualification_sha = 'b2ef2c5196a8f6777afbb25040c1a3c2f6d34b658ee9572da27c9c7d2f2c05db'
qualification = json.loads(checked(ROOT/'actual-runtime-qualification-margay-r1.json',qualification_sha))
evidence_sha = '8190de882755a37376040ebdb3a4d828d8fbe232562a417fc6b997f180c58251'
assert qualification['evidence_sha256'] == evidence_sha
evidence = json.loads(checked(qualification['evidence_path'],evidence_sha))
assert qualification['actual_HF'] == HF
assert evidence['candidate_native6_index_sha256'] == stats['candidate_units_index_sha256'] == qualification['candidate6_index_sha256']
units = json.loads(checked(ROOT/'six-candidate-native-config-coverage-item-arrays-margay-r1.json',stats['candidate_units_index_sha256']))['units']
jobs = {(unit['trial'],unit['task']):unit['actual_eval_job'] for unit in units}
assert len(jobs) == 6
for coverage in stats['coverage']:
    checked(coverage['native_done_path'],coverage['native_done_sha256'])
    item = coverage['item_scores']
    checked(item['path'],item['sha256'],item['bytes'])
for result in stats['results'].values():
    item = result['aligned_arrays']
    checked(item['path'],item['sha256'],item['bytes'])
assert len(evidence['sampling_markers']) == 6
for marker in evidence['sampling_markers']:
    assert marker['job'] == jobs[(marker['trial'],marker['task'])]
    entry = marker['original_marker_bytes_restricted_local']
    raw = json.loads(checked(entry['path'],entry['sha256'],entry['bytes']))
    assert raw['job'] == marker['job'] and raw['lines'] == marker['first3_actual_marker_lines']
    assert len(raw['lines']) == 3
    expected = {'max_tokens':8192,'temperature':0}
    if marker['trial'] != 'greedy':
        expected = {'max_tokens':8192,'temperature':0.7,'top_p':0.95,'seed':int(marker['trial'])}
    for line in raw['lines']:
        assert json.loads(line.split('TASK297_FINAL_SAMPLING_PAYLOAD ',1)[1]) == expected
assert len(qualification['six_actual_clients']) == 6
for row in qualification['six_actual_clients']:
    assert row['job'] == jobs[(row['trial'],row['task'])]
    line = row['URL_redacted_actual_CLI']
    assert f'model={HF},base_url=[REDACTED_URL],' in line
    for fragment in ['tokenizer_backend=none,','tokenized_requests=False,','num_concurrent=8,','max_length=32704','max_gen_toks=8192']:
        assert fragment in line
    encoded = re.search(r'chat_template_kwargs=base64:([^, ]+)',line).group(1)
    assert json.loads(base64.b64decode(encoded)) == {'enable_thinking':row['task']=='math500'}
job_records = {record['job']:record for record in evidence['actual_job_RPC_and_log_evidence']}
for job in jobs.values():
    record = job_records[job]
    assert record['state'] == 'succeeded'
    assert any('Using tokenizer None' in line for line in record['snippets']['tokenizer'])
    assert any('served max_model_len: 32768 (lm-eval max_length=32704)' in line for line in record['snippets']['max_length'])
serve = next(record for record in job_records.values() if record['job'].endswith('-serve'))
serve_text = json.dumps(serve['snippets'])
for fragment in ["'model_tag': '"+HF,"'model': '"+HF,"'tokenizer': '"+HF,"'dtype': 'bfloat16'","'max_model_len': 32768","'reasoning_parser': 'grug'","'data_parallel_size': 8","'enable_expert_parallel': True",'tensor_parallel_size=1','01911be34fac']:
    assert fragment in serve_text,fragment
members = {}
for archive in evidence['owner_archives']:
    for member in archive['members']:
        source = member['original_restricted_local']
        safe = member['sanitized_projection']
        raw = checked(source['path'],source['sha256'],source['bytes'])
        assert checked(safe['path'],safe['sha256'],safe['bytes']) == raw
        members[member['member']] = json.loads(raw)
assert len(members) == 8
plan = json.loads((ROOT/'launch-plan.json').read_text())
target = plan['targets']['A11']
candidate = members['candidate.json']
assert candidate['candidate'] == target['candidate']
assert candidate['sha256'] == target['candidate_sha256']
assert json.loads(checked(ROOT/'A11-candidate.json',target['candidate_sha256'])) == candidate['candidate']
assert members['source-bundles.json']['math_archive_sha256'] == plan['source_archive']['sha256']
launch = json.loads(checked(ROOT/'A11-actual-launch-margay-r1.json',qualification['actual_launch_sha256']))
assert launch['code_pin'] == plan['code_pin'] == '819f817ebbdb325b3bf8058623f3c254dc7eaadf'
assert qualification['own_actual_readiness_sha256'] == plan['actual_readiness']['sha256']
readiness = json.loads(checked(plan['actual_readiness']['path'],plan['actual_readiness']['sha256']))
assert readiness['metadata']['audit.json']['sha256'] == qualification['actual_502_audit_sha256']
event = json.loads(checked(readiness['metadata']['event.json']['path'],readiness['metadata']['event.json']['sha256']))
assert event == qualification['same_actual_HF_metadata_tokenizer_and_template_guard_provenance']['event']
assert event['hf_export'] == HF and event['completed_updates'] == 72 and event['arm'] == 'A11'

results = copy.deepcopy(stats['results'])
for result in results.values():
    contrast = result['A11_minus_existing_fresh_v01']
    result['registered_stop_triggered'] = contrast['delta_pp'] < -1.5 and contrast['ci95_pp'][1] < 0
    result.pop('stop_or_quality_decision')
    result.pop('reason')
assert not any(result['registered_stop_triggered'] for result in results.values())
output = {
    'status':'FINAL_REGISTERED_PAIRED_MATHS_A11_NO_REGISTERED_AND_STOP',
    'preregistration_sha256':stats['preregistration_sha256'],
    'conditional_statistics_sha256':stats_sha,
    'actual_runtime_qualification_sha256':qualification_sha,
    'runtime_evidence_sha256':evidence_sha,
    'reference_analysis_sha256':'c410fda8a75cf1ab2b9e2231beec8060062f9d462232e861d53cd8d45925b2df',
    'reused_reference_index_sha256':stats['reused_reference_index_sha256'],
    'runtime_independent_checks':'Six actual unit/job/source/config joins, all18 raw marker dictionaries/bytes, six CLI mode/backend/context controls, actual server HF/tokenizer/BF16/parallelism/context, eight owner member safe-projection byte equalities, wrapped candidate SHA+inner payload equality, launch/source/own72 readiness/event/audit joins, all12 done/item-array and two aligned NPZ hashes',
    'results':results,
    'statistics':stats['statistics'],
    'any_registered_stop_triggered':False,
    'interpretation':'Neither benchmark satisfies delta_pp<-1.5 AND upper95CI_pp<0. GSM has a negative interval that includes losses exceeding1.5pp; absence of this conjunction does not establish noninferiority, no loss, or release acceptance. No A3-A11 contrast added.',
    'limits':[
        'One checkpoint per condition, fixed public items and three decodes; intervals resample paired item means, not independent training runs.',
        'Reused admitted existing fresh-v01 six units; no new reference allocation or historical pooling.',
        'First18 sampling markers are not full wire HTTP or all5457 candidate payloads.',
        'Server32768/client32704 is the inherited64-token margin. Client tokenizer_backend none and tokenized_requests False.',
        'Stripped childRPC fields do not attest live worker members or effective retry policy; no runtime childZIP equality inferred.',
        'Item arrays are local native-reader projections checked against native done/coverage/headline metrics, not an independent remote per-item verifier reread.'
    ],
    'old_analysis_mutation':False,
    'model_network_storage_control_calls':0,
    'finalization_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
destination = ROOT/'paired-math-analysis-marten-r1.json'
with destination.open('x') as stream:
    json.dump(output,stream,indent=2)
    stream.write('\n')
print(json.dumps({'path':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'registered_stops':{key:value['registered_stop_triggered'] for key,value in results.items()}}))
