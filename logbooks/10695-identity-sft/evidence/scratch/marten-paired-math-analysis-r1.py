import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def metadata_records(value):
    if isinstance(value, dict):
        if {'path','sha256','bytes'} <= value.keys():
            yield value
        for child in value.values():
            yield from metadata_records(child)
    elif isinstance(value,list):
        for child in value:
            yield from metadata_records(child)


root = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b/A8a-paired-math-r2')
registration = root.parent/'bfcl-noise-r1/A8a-paired-math-preregistration-marten-r2.json'
assert digest(registration.read_bytes()) == 'ff96537805d31ee55f5f5704a7e53139229245c6dd177bac57939ef249eb6c78'
specification = json.loads(registration.read_bytes())
source = Path(specification['method']['parent_item_score_and_bootstrap'])
assert digest(source.read_bytes()) == specification['method']['parent_functions_sha256'] == 'd7bf603996af3ff9aaff5e70b5efba2e843bf0be903ba3c5221ab7a04f8cc2bc'
spec = importlib.util.spec_from_file_location('marten_paired_math_parent289',source)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
manifest_raw = (root/'all12-native-config-coverage-item-arrays-margay-r1.json').read_bytes()
assert digest(manifest_raw) == '4ef74452477d3215e0acc532b0d908065db344fbb86f7d4d134fa99b61275527'
units = json.loads(manifest_raw)['units']
assert len(units)==12
proofs = {}
for filename,expected in [
        ('local-twelve-unit-coverage-joins-marten-r1.json','f5391a7bbc0c02603c17fa8c741e16e4b6f2e95101b1be86d38eff2d7b0f919f'),
        ('actual-twelve-done-and-array-joins-marten-r1.json','cf10a998975285f2a7d912d6ee3b9777a14a68442df9a21f78156625b3b882d1'),
    ('actual-runtime-CLI-server-source-joins-marten-r1.json','acbda3460b9c5dfb37058530daa32e366150464a3365c58989ad75a6a4f1ed88')]:
    raw=(root/filename).read_bytes()
    assert digest(raw)==expected
    proofs[filename]=expected
sampling_path = Path(sys.argv[1])
sampling_raw = sampling_path.read_bytes()
assert digest(sampling_raw)==sys.argv[2]
sampling_index = json.loads(sampling_raw)
assert sampling_index['native_units_manifest_sha256']==digest(manifest_raw)
raw_sampling_path=root/'actual-runtime-qualification-r1/actual-native-workdir-hashes-and-payload-marker-observability-margay-r1.json'
raw_sampling=raw_sampling_path.read_bytes()
assert digest(raw_sampling)==sampling_index['original_native_log_query_index_sha256']=='f31c806e64002e30153f411c0cf4db5e915b7c8c0f473ee23ead0f88ffb01646'
original_sampling={record['job']:record['TASK297_FINAL_SAMPLING_PAYLOAD_retained_lines'] for record in json.loads(raw_sampling)['jobs'] if 'TASK297_FINAL_SAMPLING_PAYLOAD_retained_lines' in record}
clarification=root/'child-RPC-observability-clarification-marten-r1.json'
assert clarification.exists()
payloads_by_job = {}
checked_metadata = {}
for metadata in metadata_records(sampling_index):
    path = Path(metadata['path'])
    if str(path) in checked_metadata:
        continue
    raw=path.read_bytes()
    assert len(raw)==metadata['bytes'] and digest(raw)==metadata['sha256']
    checked_metadata[str(path)]=metadata['sha256']
    if path.suffix!='.json':
        continue
    record=json.loads(raw)
    if not isinstance(record,dict) or not {'actual_eval_job','actual_log_data_lines'} <= record.keys():
        continue
    job=record['actual_eval_job']
    assert record['actual_log_data_lines']==original_sampling[job][:3]
    payloads=[]
    for line in record['actual_log_data_lines']:
        if 'TASK297_FINAL_SAMPLING_PAYLOAD ' in line:
            payloads.append(json.loads(line.split('TASK297_FINAL_SAMPLING_PAYLOAD ',1)[1]))
    if payloads:
        assert job not in payloads_by_job, 'Multiple sampling log files for one job; audit explicitly'
        payloads_by_job[job]=payloads
assert set(payloads_by_job)=={unit['actual_eval_job'] for unit in units}
controls = []
arrays = {}
coverage = []
for unit in units:
    arm,trial,task=unit['arm'],unit['trial'],unit['task']
    payloads=payloads_by_job[unit['actual_eval_job']]
    assert len(payloads)>=3
    for payload in payloads[:3]:
        assert payload['temperature']==(0 if trial=='greedy' else 0.7)
        assert payload.get('max_tokens',payload.get('max_completion_tokens'))==8192
        if trial!='greedy':
            assert payload['top_p']==0.95 and payload['seed']==int(trial)
    controls.append(dict(arm=arm,trial=trial,task=task,actual_eval_job=unit['actual_eval_job'],first_three_final_sampling_controls=payloads[:3]))
    raw=Path(unit['item_scores']['path']).read_bytes()
    assert digest(raw)==unit['item_scores']['sha256'] and len(raw)==unit['item_scores']['bytes']
    rows=json.loads(raw)['rows']
    count=1319 if task=='gsm8k_0shot' else 500
    assert len(rows)==len({row['doc_id'] for row in rows})==count
    assert all(type(row['score']) in (int,float) and row['score'] in (0,1) for row in rows)
    assert sum(row['score'] for row in rows)==unit['n_correct']==unit['coverage']['n_correct']
    done_path=Path(unit['item_scores']['path']).with_name('native-done.json')
    assert digest(done_path.read_bytes())==unit['native_done_sha256']
    arrays[(arm,trial,task)]={row['doc_id']:row['score'] for row in rows}
    coverage.append(dict(arm=arm,trial=trial,task=task,coverage=unit['coverage'],done_sha256=unit['native_done_sha256'],item_array_sha256=unit['item_scores']['sha256']))
output=root/'paired-math-analysis-marten-r1.json'
assert not output.exists()
results={}
for task,count in [('gsm8k_0shot',1319),('math500',500)]:
    docs=sorted(arrays[('v01','greedy',task)])
    assert len(docs)==count
    assert all(set(arrays[(arm,trial,task)])==set(docs) for arm in ('A8a','v01') for trial in ('greedy','17','29'))
    matrices={arm:np.array([[arrays[(arm,trial,task)][doc] for trial in ('greedy','17','29')] for doc in docs]) for arm in ('A8a','v01')}
    differences=(matrices['A8a']-matrices['v01']).mean(axis=1)
    interval=[float(value*100) for value in analysis.bootstrap(differences,10000,297)]
    delta=float(differences.mean()*100)
    aligned=root/f'{task}-aligned-three-decode-marten-r1.npz'
    assert not aligned.exists()
    np.savez(aligned,doc_ids=np.array(docs),A8a=matrices['A8a'],v01=matrices['v01'],item_mean_difference=differences)
    results[task]=dict(items=count,thinking=task=='math500',metric='exact_match' if task=='gsm8k_0shot' else 'accuracy',filter='flexible-extract' if task=='gsm8k_0shot' else 'none',decode_order=['greedy','17','29'],
        counts={arm:matrix.sum(axis=0).astype(int).tolist() for arm,matrix in matrices.items()},
        item_mean_accuracy_pct={arm:float(matrix.mean()*100) for arm,matrix in matrices.items()},
        A8a_minus_fresh_v01=dict(delta_pp=delta,ci95_pp=interval,better=int((differences>0).sum()),worse=int((differences<0).sum()),ties=int((differences==0).sum())),
        registered_stop_line='delta_pp < -1.5 AND upper95CI_pp < 0',trips_registered_stop_line=bool(delta < -1.5 and interval[1] < 0),
        aligned_arrays=dict(path=str(aligned),sha256=digest(aligned.read_bytes()),bytes=aligned.stat().st_size))
report=dict(status='COMPLETE_PAIRED_THREE_DECODE_ANALYSIS_WITH_EXPLICIT_RUNTIME_CAVEATS',preregistration_sha256=digest(registration.read_bytes()),
    native_unit_manifest_sha256=digest(manifest_raw),independent_proofs=proofs,actual_sampling_index=dict(path=str(sampling_path),sha256=digest(sampling_raw)),
    verified_sampling_log_files=checked_metadata,first_three_actual_final_payloads_per_unit=controls,coverage=coverage,results=results,
    child_RPC_observability_clarification=dict(path=str(clarification),sha256=digest(clarification.read_bytes())),
    statistics='Exact parent289.bootstrap: average three within-item candidate-reference decode differences; NumPy default_rng297 independently per benchmark,10000 N-item resamples,percentiles2.5/97.5; no task/decode/run selection, no historical pooling, no multiplicity adjustment.',
    runtime_caveats=['Configured/server context32768; effective client32704 from frozen64-token margin, with unchanged8192 generation budget.',
        'Actual client tokenizer_backend=none,tokenized_requests=False,Using tokenizer None; server tokenizers follow each bound HF. Public --tokenizer label in wrapper is not a loaded client tokenizer revision.',
        'Stripped legacy/federated child RPC readback reports preemption100/timeout0/constraints[] with empty entrypoint/bundle/workdir. These are NOT established effective runtime policy; archived Iris signature default1000, stale docstring100, projection origin unresolved. Actual attempts retained; no policy repair or claim every child0/0.',
        'Legacy child bundle_id fields empty; do not equate them to local rebuilt f001. Verified rootea871/executionarchive2f0/sourceclient+runner and observed model/CLI/finalcontrols; direct actual child workdir member proof is a separately qualified limit.',
        'Sampling evidence is the first three actual finalized-control log markers per unit, not full per-request content or wireHTTP capture.'],
    interpretation_limit='Fixed public items and three decoding settings for one checkpoint per condition; no replicated-training, universal capability, full source-byte identity or release acceptance. Statistical stop result does not authorize another experiment or overwrite BFCL/identity records.',
    script_sha256=digest(Path(__file__).read_bytes()),model_GPU_network_storage_calls=0)
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(path=str(output),sha256=digest(output.read_bytes()),results=results),indent=2))
