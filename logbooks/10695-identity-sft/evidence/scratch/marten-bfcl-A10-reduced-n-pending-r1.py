import ast
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
HERE = ROOT/'bfcl-noise-r1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    raw = Path(path).read_bytes()
    assert digest(raw) == expected, path
    return json.loads(raw)


amendment_sha = 'f44554ba02340cfcd8a787b0225fc7857e1558e084fb2c76d388b4a326cdfe91'
checked(HERE/'BFCL-infra-exhaustion-analysis-addendum-marten-r1.json',amendment_sha)
registration_sha = 'ad360b5c06c6726c7641024092b8d8955bead17a75e5d5605f9253aa55da2f48'
checked(HERE/'A10-run-comparison-preregistration-marten-r1.json',registration_sha)
frozen_path = HERE/'A8-joint-run-bootstrap-extension-marten-r1.json'
frozen_sha = 'f2c3f64fa85c511c8902ad12a5e1b1dae3dc5479b6b3ad8a285e426c7ea2cdd1'
frozen = checked(frozen_path,frozen_sha)
replay_path = Path('/mnt/storage/karan-scratch/marten-bfcl-a8-extension-r1.py')
source = replay_path.read_bytes()
assert digest(source) == frozen['source_script_sha256']
module = ast.parse(source)
position = next(index for index,node in enumerate(module.body) if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='output' for target in node.targets))
namespace = {'__name__':'frozen_A8_prefix_replay','__file__':str(replay_path)}
exec(compile(ast.Module(body=module.body[:position],type_ignores=[]),str(replay_path),'exec'),namespace)
assert namespace['summaries'] == frozen['summaries'] and namespace['contrasts'] == frozen['contrasts']
canonical = namespace['generator']
interval = namespace['interval']
comparators = {
    'fresh_v01': namespace['comparators']['fresh_v01'],
    'A3': namespace['comparators']['A3'],
    'A8a': (namespace['fractions']['A8a'],namespace['draws']['A8a']),
}
for iteration in range(10000):
    for cluster in range(5):
        canonical.choice(range(5))
a10_start = canonical.getstate()
generator = random.Random()
generator.setstate(a10_start)
index_path = ROOT/'A10-recovered-BFCL-five-r1/A10-run-index-20261030T195858.346212Z.json'
index_sha = '285a1d199e53aaec3131827d60a0f89677157ffafec8681ec29a288f0fa16cee'
index = checked(index_path,index_sha)
assert index['logical_planned']==5 and index['complete123']==4 and index['exhausted_final_R2']==1
assert index['ALL_CHAINS_FULFILLED_OR_EXHAUSTED'] and not index['ALL_FIVE_COMPLETE']
assert index['reduced_n_analysis_addendum_sha256']==amendment_sha and index['primary_preregistration_sha256']==registration_sha
physical = {row['job']:row for row in index['physical_runs']}
assert len(physical)==len(index['physical_runs'])==14 and all(row['SDK_terminal'] for row in physical.values())
assert len(index['runs'])==5 and {row['original_slot'] for row in index['runs']}==set(range(1,6))
admitted = []
exhausted = []
for logical in index['runs']:
    slot = logical['original_slot']
    chain = sorted([row for row in physical.values() if row['original_slot']==slot],key=lambda row:row['replacement_number'])
    assert [row['replacement_number'] for row in chain]==list(range(len(chain))) and len(chain)<=3
    assert logical['all_physical_jobs_retained']==[row['job'] for row in chain]
    assert logical['job']==chain[-1]['job']
    valid = [row for row in chain if row['state']=='COMPLETE123']
    assert len(valid)<=1
    for physical_row in chain:
        launch = checked(physical_row['launch'],physical_row['launch_sha256'])
        assert launch['job']==physical_row['job'] and launch['HF']==index['source_HF']
        assert physical_row['readiness_sha256']=='df7db2d3393d2118a189f7d93b8a4b622657df7b8552fb687d57776fe849e0ba'
    if logical['logical_chain_status']=='EXHAUSTED_FINAL_R2_NO_FULL123':
        assert not valid and logical['score'] is None and logical['replacement_number']==2
        assert logical['original_slot']==2
        exhausted.append(logical)
        continue
    assert logical['logical_chain_status']=='FULFILLED_FULL123' and len(valid)==1 and valid[0]['job']==logical['job']
    row = valid[0]
    report = checked(row['report'],row['report_sha256'])
    array = checked(row['array'],row['array_sha256'])
    assert array['score_sha256']==row['report_sha256'] and array['job']==report['job']==row['job']
    assert report['HF']==row['HF']==index['source_HF']
    assert report['method_pin']=='caba0fa38b780af42da54af2f188ba5bfcc28501'
    assert report['readiness_sha256']==row['readiness_sha256']
    assert report['native_record_sha256']==row['native_record_sha256']
    assert report['source_model_yaml_sha256']=='e3877fb0b29f537a395826923c447a65675820a864696dc1fc0935dcb1dbbb0c'
    assert report['native_model_config_sha256']=='ac5846334f9f9de49fb7cb3bdddbac007925c3a5993ad8128755a527c20e8b7c'
    rewards = {entry['doc_id']:entry['reward'] for entry in array['rows']}
    assert len(array['rows'])==len(rewards)==123 and set(rewards)==namespace['task_ids']
    assert all(type(value) in (int,float) and math.isfinite(value) and value in (0,1) for value in rewards.values())
    count = int(sum(rewards.values()))
    assert report['n']==123 and count==row['score']==row['observed_correct']==report['correct']
    assert math.isclose(report['mean_pct'],100*count/123,abs_tol=1e-12,rel_tol=0)
    coverage = report['native_coverage']['bfcl_parity']
    assert all(coverage[field]==123 for field in ['n_benchmark','n_attempted','n_scored'])
    admitted.append(dict(slot=slot,run_id=row['job'],correct=count,report_sha256=row['report_sha256'],array_sha256=row['array_sha256'],native_record_sha256=row['native_record_sha256'],replacement_number=row['replacement_number']))
admitted.sort(key=lambda row:row['run_id'])
fractions = [row['correct']/123 for row in admitted]
assert len(fractions)==4
draws = [statistics.fmean(generator.choice(fractions) for cluster in range(4)) for iteration in range(10000)]
for iteration in range(10000):
    for cluster in range(5):
        canonical.choice(range(5))
summary = dict(planned=5,n=4,exhausted_infra_no_score=1,pending=0,physical_workflows=14,runs=admitted,
               mean_percent=100*statistics.fmean(fractions),sample_SD_percent=100*statistics.stdev(fractions),
               min_max_percent=[100*min(fractions),100*max(fractions)],unadjusted_mean_CI95_percent=[100*value for value in interval(draws)])
contrasts = {f'A10_minus_{label}':dict(delta_pp=100*(statistics.fmean(fractions)-statistics.fmean(values)),
              unadjusted_CI95_pp=[100*value for value in interval([new-old for new,old in zip(draws,old_draws,strict=True)])]) for label,(values,old_draws) in comparators.items()}
assert digest(frozen_path.read_bytes())==frozen_sha
output = dict(status='A10_COMPLETE_CHAINS_N4_STATISTICS_CONDITIONAL_ACTUAL_NATIVE_RAW_QUALIFICATION_PENDING',
               index_path=str(index_path),index_sha256=index_sha,preregistration_sha256=registration_sha,reduced_n_addendum_sha256=amendment_sha,
               frozen_A8_analysis_sha256=frozen_sha,all_old_26_ref_A6_A8_summaries_and_intervals_exactly_replayed=True,
               reserved_A9_fixed_n5_block_advanced_without_outcomes=True,A10_actual_n4_from_cloned_reserved_block_start=True,
               A10_reserved_fixed_n5_canonical_block_advanced_for_future_A11=True,
               A10_block_start_state_sha256=digest(json.dumps(a10_start).encode()),A11_canonical_block_start_state_sha256=digest(json.dumps(canonical.getstate()).encode()),
               summary=summary,contrasts=contrasts,exhausted_slot_numbers=[2],
               primary=['A10_minus_fresh_v01'],secondary=['A10_minus_A8a','A10_minus_A3'],
               final_interpretation=None,actual_native_scored_samples_and_uniform_eval_model_config_qualification='PENDING_RAW_RECORD_AND_ALL123_SAMPLE_READBACK',
               statistics='f445: actualn4 choices eachiteration in cloned A10 reserved state, PythonRandom297 frozen-prefix replay,10000draws/indices249,9749; sharedcomparators/noitem/repeatpairing/oldpool/newcontrast',
               limits=['Complete-score survivors only; exhaustion need not be missing-at-random.','Four evaluation clusters/one trained checkpoint; unadjusted intervals, no release or universal causal acceptance.','No valid score rerun, partial repair, missing-as-zero or infra-attempt resampling.','Native raw grading/config qualification remains pending; this file is conditional.'],
               source_script_sha256=digest(Path(__file__).read_bytes()),model_network_storage_control_calls=0)
destination = HERE/'A10-run-bootstrap-pending-native-marten-r1.json'
with destination.open('x') as stream:
    json.dump(output,stream,indent=2)
    stream.write('\n')
print(json.dumps({'path':str(destination),'sha256':digest(destination.read_bytes()),'summary':summary,'contrasts':contrasts},indent=2))
