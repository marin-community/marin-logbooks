import hashlib
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path

import yaml

ROOT=Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path,expected):
    raw=Path(path).read_bytes()
    assert sha(raw)==expected, str(path)
    return json.loads(raw)

def native_reward(sample):
    metrics=sample['metrics']
    if isinstance(metrics,list):
        values=[value for name,value in metrics if name=='reward']
    else:
        values=[metrics['reward']] if 'reward' in metrics else []
    assert len(values)<=1
    return values[0] if values else None

spec_path=ROOT/'bfcl-noise-r1/reference-extension-spec-marten-r1.json'
spec_sha='4e449bc78bb7a0f5ac6502d7d72581c1f888d03cc13701ec90b1caf72bfd1cd8'
spec=checked(spec_path,spec_sha)
primary_path=ROOT/'bfcl-noise-r1/primary-run-bootstrap-marten-r1.json'
primary_sha='06f71ba85ac51e23471bba15bcb6198b2a02528f74af302b986fae39c466d9fa'
frozen=checked(primary_path,primary_sha)
old_spec=importlib.util.spec_from_file_location('frozen_primary_reader','/mnt/storage/karan-scratch/marten-bfcl-run-bootstrap-r3.py')
old=importlib.util.module_from_spec(old_spec)
old_spec.loader.exec_module(old)
assert sha(Path(old.__file__).read_bytes())==frozen['source_script_sha256']
index=checked(frozen['index_path'],frozen['index_sha256'])
failures=old.audit_expanded_physical_chain(index)
scored,pending,admitted=old.audit(index)
assert not pending and admitted==frozen['admitted_runs']
arms={arm:[count for run_id,count in scored[arm]] for arm in old.ARMS}
assert old.bootstrap(arms)==frozen['primary_analysis']

ref_index_path=ROOT/'bfcl-v01-reference5-r1/reference-run-index-20261030T082646.869642Z.json'
ref_index_sha='69b3a03e8880320580ca43a445f4c9d4b97d9e2245bff9819a9bd28d932ce351'
ref_index=checked(ref_index_path,ref_index_sha)
assert ref_index['ALL_FIVE_COMPLETE'] and ref_index['complete123']==ref_index['required_fresh_runs']==5
assert ref_index['physical_workflow_count']==9
assert not ref_index['historical34_pooled'] and not ref_index['old_r5_pooled']
assert ref_index['frozen26_arm_analysis_unchanged_sha256']==primary_sha
physical={row['run_id']:row for row in ref_index['physical_reference_runs']}
assert len(physical)==len(ref_index['physical_reference_runs'])==9
assert len(ref_index['reference_runs'])==5 and {row['reference_run_index'] for row in ref_index['reference_runs']}==set(range(1,6))
source_ids=checked(ref_index['task_ids_only_join_source'],ref_index['task_ids_only_join_source_sha256'])
task_ids={row['doc_id'] for row in source_ids['rows']}
assert len(task_ids)==123
source_config_path=ROOT/'bfcl-v01-reference5-r1/model-v01.yaml'
yaml_sha='d24f047a6c5c908b279a312d18cbb6f9d0969ac56b8f54c08e5d1400f1ca6730'
assert sha(source_config_path.read_bytes())==yaml_sha
new_yaml=yaml.safe_load(source_config_path.read_bytes())
old_yaml=yaml.safe_load((ROOT/'bfcl-ablation-r1/model-A0.yaml').read_bytes())
assert {key:value for key,value in new_yaml.items() if key not in ['name','location']}=={key:value for key,value in old_yaml.items() if key not in ['name','location']}
source_proof_sha='ec36aa1bfd37b173a3612ecaba9e273ea7220c4990f748b3e5024f689d595bd5'
checked(ROOT/'bfcl-v01-reference5-r1/plain-v01-binding-and-source-proof-r1.json',source_proof_sha)
location='s3://marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a'
method='caba0fa38b780af42da54af2f188ba5bfcc28501'
native_config_sha='7d89a9de6f0c478dcea6a277dc09f83b719e6773dbc350e89cb63e9d44d3d63b'
native_configs=[]
partial_evidence=[]
all_launches={}
for row in physical.values():
    launch=checked(row['launch'],row['launch_sha256'])
    assert launch['job']==row['run_id'] and launch['reference_run_index']==row['reference_run_index']
    assert launch['model_yaml_sha256']==row['model_config_sha256']==yaml_sha
    assert launch['HF']==row['HF']==location and launch['method_pin']==row['method_pin']==method
    all_launches[row['run_id']]=launch
    native=checked(row['native_record'],row['native_record_sha256'])
    assert native['status']=='succeeded' and native['model']['location']==location
    assert native['model']['config_digest']==native_config_sha
    native_configs.append(native['model']['config'])
    assert native['coverage']['bfcl_parity']['n_benchmark']==123
    if row['state']=='COMPLETE123':
        continue
    assert row['state']=='PARTIAL123_NOT_ADMISSIBLE' and row['score'] is None
    inspection=checked(row['partial_inspection'],row['partial_inspection_sha256'])
    samples=checked(row['all_samples'],row['all_samples_sha256'])
    assert len(samples)==123 and {sample['doc_id'] for sample in samples}==task_ids
    missing=[]
    for sample in samples:
        reward=native_reward(sample)
        if reward is not None:
            assert type(reward) in (int,float) and math.isfinite(reward) and reward in (0,1)
            continue
        detail=json.loads(sample['grading']['detail'])
        assert detail['scored'] is False and detail['rewards']=={}
        assert detail['error']['type']=='EnvironmentStartTimeoutError' and detail['error']['category']=='infrastructure'
        missing.append(sample['doc_id'])
    assert len(missing)==row['missing_rewards']==inspection['missing_rewards']
    assert 123-len(missing)==row['scored_binary_rewards']==inspection['scored_binary_rewards']
    partial_evidence.append(dict(run_id=row['run_id'],logical_slot=row['reference_run_index'],missing=len(missing),
        observed_binary=123-len(missing),all_missing_actual_infrastructure_grading=True,
        samples_sha256=row['all_samples_sha256'],not_scored_or_imputed_for_statistics=True))
assert len(partial_evidence)==4 and sorted(row['missing'] for row in partial_evidence)==[4,5,7,9]
assert all(config==native_configs[0] for config in native_configs)
reference=[]
for row in ref_index['reference_runs']:
    physical_row=physical[row['run_id']]
    assert {key:value for key,value in row.items() if key not in ['logical_slot','full_physical_chain_retained']}==physical_row
    chain=row['full_physical_chain_retained']
    slot=row['reference_run_index']
    expected_chain=[item['run_id'] for item in ref_index['physical_reference_runs'] if item['reference_run_index']==slot]
    assert chain==expected_chain and chain[-1]==row['run_id']
    if row['replacement_number']:
        assert row['replacement_number']==1 and len(chain)==2
        failed=physical[chain[0]]
        assert failed['state']=='PARTIAL123_NOT_ADMISSIBLE' and row['replaces_incomplete_job']==chain[0]
        assert all_launches[row['run_id']]['replaces_incomplete_job']==chain[0]
    else:
        assert slot==4 and len(chain)==1
    array=checked(row['array'],row['array_sha256'])
    report=checked(row['report'],row['report_sha256'])
    assert array['score_sha256']==row['report_sha256']
    assert array['historical34_rewards_included'] is False
    assert array['run_id']==report['run_id']==row['run_id']
    assert array['results_uri']==report['results_uri']==row['results_uri']
    assert len(array['rows'])==123
    assert all(set(entry)=={'doc_id','reward'} for entry in array['rows'])
    rewards={entry['doc_id']:entry['reward'] for entry in array['rows']}
    assert set(rewards)==task_ids and all(type(value) in (int,float) and math.isfinite(value) and value in (0,1) for value in rewards.values())
    correct=int(sum(rewards.values()))
    assert correct==row['score']==row['correct']==report['correct'] and report['n']==123
    assert math.isclose(report['mean_pct'],correct*100/123,abs_tol=1e-12)
    assert report['native_record_sha256']==row['native_record_sha256']
    assert report['source_model_yaml_sha256']==yaml_sha and report['native_model_config_sha256']==native_config_sha
    assert report['method_pin']==method and not report['historical34_pooled'] and not report['old_r5_pooled']
    samples_path=Path(row['native_record']).with_name('samples.actual.readonly.json')
    samples_raw=samples_path.read_bytes()
    samples=json.loads(samples_raw)
    assert len(samples)==123 and {sample['doc_id']:native_reward(sample) for sample in samples}==rewards
    assert all(json.loads(sample['grading']['detail'])['scored'] is True for sample in samples)
    reference.append(dict(run_id=row['run_id'],logical_slot=slot,correct=correct,array_sha256=row['array_sha256'],
        report_sha256=row['report_sha256'],native_record_sha256=row['native_record_sha256'],
        native_samples_sha256=sha(samples_raw)))
reference.sort(key=lambda row:row['run_id'])
assert sorted(row['correct'] for row in reference)==[31,32,32,34,41]

fractions={arm:[count/123 for count in arms[arm]] for arm in old.ARMS}
generator=random.Random(297)
draws={arm:[] for arm in old.ARMS}
old_contrasts={arm:[] for arm in old.ARMS[1:]}
for draw in range(10000):
    means={}
    for arm in old.ARMS:
        means[arm]=statistics.fmean(generator.choice(fractions[arm]) for cluster in range(old.EXPECTED[arm]))
        draws[arm].append(means[arm])
    for arm in old.ARMS[1:]:
        old_contrasts[arm].append(means[arm]-means['A0'])
for arm in old.ARMS:
    assert [100*value for value in old.interval(draws[arm])]==frozen['primary_analysis']['per_arm'][arm]['unadjusted_mean_CI95_percent']
for arm in old.ARMS[1:]:
    assert [100*value for value in old.interval(old_contrasts[arm])]==frozen['primary_analysis']['contrasts'][arm+'_minus_A0']['unadjusted_CI95_pp']
reference_fractions=[row['correct']/123 for row in reference]
reference_draws=[statistics.fmean(generator.choice(reference_fractions) for cluster in range(5)) for draw in range(10000)]
reference_mean=statistics.fmean(reference_fractions)
summary=dict(n=5,run_order='Lexical actual scored run_id',runs=reference,
    mean_percent=reference_mean*100,sample_SD_percent=100*statistics.stdev(reference_fractions),
    min_max_percent=[100*min(reference_fractions),100*max(reference_fractions)],
    unadjusted_mean_CI95_percent=[100*value for value in old.interval(reference_draws)])
contrasts={arm+'_minus_fresh_v01':dict(delta_pp=100*(statistics.fmean(fractions[arm])-reference_mean),
    unadjusted_CI95_pp=[100*value for value in old.interval([arm_value-ref_value for arm_value,ref_value in zip(draws[arm],reference_draws,strict=True)])]) for arm in old.ARMS}
assert sha(primary_path.read_bytes())==primary_sha
output=dict(status='COMPLETE_FIVE_FRESH_REFERENCE_RUN_BOOTSTRAP_EXTENSION',
    extension_spec_sha256=spec_sha,original_spec_sha256=old.SPEC_SHA256,
    frozen26_primary_sha256=primary_sha,frozen26_primary_unchanged=True,
    reference_index_path=str(ref_index_path),reference_index_sha256=ref_index_sha,
    reference_physical_workflows=9,reference_complete_logical_slots=5,preserved_partial_reference_workflows=partial_evidence,
    complete_reference_admission='All5 exact123 unique binary rewards, native samples/report/record/launch/config hashes and replacement chains joined',
    original41_physical_and26_logical_admission_reproduced=True,
    statistics=dict(seed=297,draws=10000,CI_order_statistic_indices=[249,9749],multiplicity_adjustment='none',
        independent_run_resampling=True,no_repeat_number_pairing=True,task_resampling=False,
        frozen_arm_stream_and_all_original_intervals_reproduced_exactly=True,
        reference_draws='Continue the same PRNG stream after the original10000 A0-A4 draw iterations; independent5-run reference resamples; shared reference draw across contrasts'),
    reference=summary,arm_means_preserved={arm:frozen['primary_analysis']['per_arm'][arm] for arm in old.ARMS},
    arm_minus_new_reference=contrasts,
    historical34_not_pooled=True,old_r5_not_pooled=True,partial_rewards_not_imputed_or_pooled=True,
    task_ID_join_does_not_use_historical_rewards_for_statistics=True,
    source_model_yaml_sha256=yaml_sha,source_proof_sha256=source_proof_sha,native_reference_config_sha256=native_config_sha,
    effective_temperature_1_no_seed='Source-derived existing policy; not a newly captured HTTP request',
    limitations=spec['limitations'],script_sha256=sha(Path(__file__).read_bytes()),model_or_remote_calls=0)
path=ROOT/'bfcl-noise-r1/fresh-v01-reference-extension-marten-r1.json'
assert not path.exists()
path.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(path=str(path),sha256=sha(path.read_bytes()),reference=summary,contrasts=contrasts),indent=2))
