import ast
import hashlib
import json
import random
import statistics
from pathlib import Path


ROOT = Path('/home/karan/Dropbox/marin-notes/snowball-identity/path-b')
HERE = ROOT/'bfcl-noise-r1'
FROZEN = {
    'primary-run-bootstrap-marten-r1.json':'06f71ba85ac51e23471bba15bcb6198b2a02528f74af302b986fae39c466d9fa',
    'fresh-v01-reference-extension-marten-r1.json':'e6a6b5e223acfc945330d9ac5d5bba5a82863bb61e66754048d84526ba5b6df0',
    'A6-run-bootstrap-extension-marten-r1.json':'1dbd6ba061ab8525b55ec0b8a7592c652730dac5cc0613846b5c7407309cad3e',
    'A8-joint-run-bootstrap-extension-marten-r1.json':'f2c3f64fa85c511c8902ad12a5e1b1dae3dc5479b6b3ad8a285e426c7ea2cdd1',
    'A9-run-bootstrap-extension-marten-r1.json':'7eefff9eaf93c75b6c9b19c430296057e7e2cdd969ff78e08ed1e117bbd356de',
    'A10-run-bootstrap-extension-marten-r1.json':'37bc83d8411d62748be16e546fa70ec9aa73bc804fb302b3ad14770ceedcd62c',
    'A11-run-bootstrap-extension-marten-r1.json':'b42e9fd2fec0fbc896a82c2fae89623b2951e5dbf7bdfdd31c2a51d5b4ddba71',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    raw = Path(path).read_bytes()
    assert digest(raw)==expected,path
    return json.loads(raw)


def state_digest(generator):
    return digest(json.dumps(generator.getstate()).encode())


frozen = {name:checked(HERE/name,sha) for name,sha in FROZEN.items()}
registration_sha = '66f81727ef745fd63b2decd0a32702f971b83262ee5090af97c327488946bd22'
checked(HERE/'A11-run-comparison-preregistration-marten-r1.json',registration_sha)
amendment_sha = 'f44554ba02340cfcd8a787b0225fc7857e1558e084fb2c76d388b4a326cdfe91'
checked(HERE/'BFCL-infra-exhaustion-analysis-addendum-marten-r1.json',amendment_sha)
a8 = frozen['A8-joint-run-bootstrap-extension-marten-r1.json']
replay_path = Path('/mnt/storage/karan-scratch/marten-bfcl-a8-extension-r1.py')
source = replay_path.read_bytes()
assert digest(source)==a8['source_script_sha256']
module = ast.parse(source)
position = next(index for index,node in enumerate(module.body) if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='output' for target in node.targets))
namespace = {'__name__':'frozen_A8_prefix_replay','__file__':str(replay_path)}
exec(compile(ast.Module(body=module.body[:position],type_ignores=[]),str(replay_path),'exec'),namespace)
assert namespace['summaries']==a8['summaries'] and namespace['contrasts']==a8['contrasts']
canonical = namespace['generator']
interval = namespace['interval']
comparators = {
    'fresh_v01':namespace['comparators']['fresh_v01'],
    'A3':namespace['comparators']['A3'],
    'A8a':(namespace['fractions']['A8a'],namespace['draws']['A8a']),
}
draws = {}
fractions = {}
states = {}
for arm in ['A9','A10','A11']:
    record = frozen[f'{arm}-run-bootstrap-extension-marten-r1.json']
    assert state_digest(canonical)==record[f'{arm}_block_start_state_sha256']
    states[f'{arm}_start']=state_digest(canonical)
    runs = record['summary']['runs']
    assert [row['run_id'] for row in runs]==sorted(row['run_id'] for row in runs)
    fractions[arm] = [row['correct']/123 for row in runs]
    analysis_generator = random.Random()
    analysis_generator.setstate(canonical.getstate())
    draws[arm] = [statistics.fmean(analysis_generator.choice(fractions[arm]) for cluster in range(len(runs))) for iteration in range(10000)]
    for iteration in range(10000):
        for cluster in range(5):
            canonical.choice(range(5))
    summary = record['summary']
    assert summary['n']==len(runs)
    assert summary['mean_percent']==100*statistics.fmean(fractions[arm])
    assert summary['sample_SD_percent']==100*statistics.stdev(fractions[arm])
    assert summary['min_max_percent']==[100*min(fractions[arm]),100*max(fractions[arm])]
    assert summary['unadjusted_mean_CI95_percent']==[100*value for value in interval(draws[arm])]
    for contrast_name,contrast in record['contrasts'].items():
        label = contrast_name.removeprefix(f'{arm}_minus_')
        values,old_draws = comparators[label]
        assert contrast==dict(delta_pp=100*(statistics.fmean(fractions[arm])-statistics.fmean(values)),unadjusted_CI95_pp=[100*value for value in interval([new-old for new,old in zip(draws[arm],old_draws,strict=True)])])
    end_key = {'A9':'A10_canonical_block_start_state_sha256','A10':'A11_canonical_block_start_state_sha256','A11':'after_A11_canonical_block_state_sha256'}[arm]
    assert state_digest(canonical)==record[end_key]
    states[f'{arm}_canonical_end']=state_digest(canonical)
assert frozen['A11-run-bootstrap-extension-marten-r1.json']['pending_registered_contrast']=={'A11_minus_A9':'A9_LOGICAL_CHAINS_NOT_FINAL_NO_PARTIAL_MEAN_OR_IMPUTATION'}
contrast = dict(delta_pp=100*(statistics.fmean(fractions['A11'])-statistics.fmean(fractions['A9'])),unadjusted_CI95_pp=[100*value for value in interval([new-old for new,old in zip(draws['A11'],draws['A9'],strict=True)])])
for name,sha in FROZEN.items():
    assert digest((HERE/name).read_bytes())==sha
output = dict(
    status='FINAL_REGISTERED_A11_MINUS_A9_CONTRAST_APPEND_ONLY',
    preregistration_sha256=registration_sha,reduced_n_addendum_sha256=amendment_sha,
    frozen_analysis_files_unchanged=FROZEN,
    actual_A9_native615_qualification_sha256=frozen['A9-run-bootstrap-extension-marten-r1.json']['actual_native_readback_sha256'],
    actual_A11_native615_qualification_sha256=frozen['A11-run-bootstrap-extension-marten-r1.json']['actual_native_readback_sha256'],
    sample_sizes=dict(A9=5,A11=5),physical_workflows=dict(A9=12,A11=8),
    means_percent={arm:100*statistics.fmean(fractions[arm]) for arm in ['A9','A11']},
    contrast={'A11_minus_A9':contrast},
    exact_replay=dict(all_old_26_ref_A6_A8_summaries_and_CIs=True,A9_A10_A11_summaries_and_all_previously_final_contrasts=True,state_chain=states),
    statistics='Independent per-arm run bootstrap, PythonRandom297,10000 draws; lexical actual scored run IDs. Exact frozen prefix through A8 followed by fixed n5 A9, canonical n5 A10 reservation and n5 A11. Actual A10 n4 replayed only in its cloned state. A11 draw j minus independent A9 draw j; no pairing by repeat number or task resampling. Unadjusted percentile endpoints at indices249/9749.',
    limits=['Five evaluation runs per arm on one trained checkpoint each; unadjusted few-cluster interval.','Full logical coverage after authorized infrastructure replacements; all failed physical attempts retained, no best-of or missing-as-zero.','A11 and A9 have different realized identity/replay doses, consumed cutoffs and packing. This contrast does not isolate a pure causal ON-share effect.','No A11-minus-A10 or other new contrast, equivalence test, release acceptance or A9 identity inference.'],
    source_script_sha256=digest(Path(__file__).read_bytes()),model_network_storage_control_calls=0,
)
destination = HERE/'A11-minus-A9-registered-contrast-final-marten-r1.json'
with destination.open('x') as stream:
    json.dump(output,stream,indent=2)
    stream.write('\n')
print(json.dumps(dict(path=str(destination),sha256=digest(destination.read_bytes()),contrast=contrast),indent=2))
