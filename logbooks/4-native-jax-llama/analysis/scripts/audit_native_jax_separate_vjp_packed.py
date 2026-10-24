"""Audit retained separate-capture artifacts; no execution/remote access."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import tarfile

root = Path('/private/tmp')
r = root / 'native-jax-bf16-splash-15b-separate-vjp-packed-results'
archive = root / 'native-jax-bf16-splash-15b-separate-vjp-packed-diagnostics.tar.gz'
load = lambda name: json.loads((r / name).read_text())
p = load('provenance.json')
c = load('controls.json')
o = load('case-outcomes.json')
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive_sha == '180831fc4350dd277069a4daf321affc89391a479a62a17cc91882b485192b8e'
with tarfile.open(archive) as tf:
    members = tf.getmembers()
    assert len(members) == 48
    for member in members:
        path = Path(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts
        assert tf.extractfile(member).read() == (r / path).read_bytes()
for item in o['files']:
    data = (r / item['file']).read_bytes()
    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
source_matches = {}
for filename, sha in p['source_file_sha256'].items():
    actual = hashlib.sha256(subprocess.check_output(['git', 'show', p['source_commit'] + ':' + filename], cwd='/private/tmp/marin-native-jax-splash-mask-sharding')).hexdigest()
    source_matches[filename] = actual == sha
assert all(source_matches.values())
original = json.loads((root / 'native-jax-bf16-splash-15b-block-results/provenance.json').read_text())
assert p['parameter_sha256'] == original['parameter_sha256']
assert p['batch_sha256'] == original['batch_sha256']
assert p['parameter_count'] == 1531037696
frozen_operands = {k: v['sha256'] == original['source_operands'][k]['sha256'] for k, v in p['source_operands'].items()}
assert len(frozen_operands) == 5 and all(frozen_operands.values())
assert all(x['exact'] and x['expected'] == x['actual'] for x in p['source_operand_hashes_vs_original'].values())
assert p['capture_controls_exact'] and p['source_cotangents_nonzero']
assert p['canonical_master_roundtrip'] and all(v['named']['exact'] and v['native']['exact'] for v in p['canonical_master_roundtrip'].values())
hlo = []
finite_record_numbers = True

def walk(value):
    global finite_record_numbers
    if isinstance(value, dict):
        if {'saved_bytes', 'file', 'truncated'} <= value.keys():
            hlo.append(value)
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
    elif isinstance(value, float):
        finite_record_numbers &= math.isfinite(value)

for f in r.glob('*.json'):
    walk(json.loads(f.read_text()))
assert finite_record_numbers
for h in hlo:
    data = (r / h['file']).read_bytes()
    assert not h['truncated'] and len(data) == h['full_bytes'] == h['saved_bytes']
    assert hashlib.sha256(data).hexdigest() == h.get('sha256_saved', h.get('sha256'))
assert len(hlo) == 34
assert sum(x['saved_bytes'] for x in hlo) == o['hlo_bytes']
stages = {}
all_results = {}
for stage in ['full', 'prefix', 'tail', 'composed']:
    d = load(f'last-{stage}.json')
    all_results[stage] = d
    left, right = d['named'], d['native']
    keys = ['shape', 'dtype', 'partition_spec', 'sha256']
    assert all(all(a[k] == b[k] for k in keys) for a, b in zip(left['arguments'], right['arguments'], strict=True))
    assert all(left['cotangent'][k] == right['cotangent'][k] for k in keys)
    assert all(all(a[k] == b[k] for k in keys[:-1]) for a, b in zip(left['results'], right['results'], strict=True))
    assert d['common_bf16_operands_fp32_masters_and_physical_specs_exact']
    assert len(left['result_order']) == len(left['results']) == 11
    assert [x['dtype'] for x in left['results']] == ['bfloat16', 'bfloat16'] + ['float32'] * 9
    comparisons = dict(zip(left['result_order'], d['comparisons']['named_vs_native'], strict=True))
    stages[stage] = {'result_order': left['result_order'], 'named_native_exact_outputs': [name for name, value in comparisons.items() if value['exact']], 'named_native_nonexact_outputs': {name: {**value, 'relative_l2_percent': 100 * value['relative_l2']} for name, value in comparisons.items() if not value['exact']}, 'common_operand_hashes_shapes_dtypes_partitions_exact': True, 'result_shapes_dtypes_partitions_exact': True}
full_composed_hash_matches = {}
for variant in ['named', 'native', 'fp32']:
    full_composed_hash_matches[variant] = all(a['sha256'] == b['sha256'] for a, b in zip(all_results['full'][variant]['results'], all_results['composed'][variant]['results'], strict=True))
assert all(full_composed_hash_matches.values())
assert all(x['exact'] for values in c['full_vs_composed'].values() for x in values)
source_graphs = ['source-scan', 'capture-without-duplicate-full', 'source-full-separate', 'normal-hidden', 'normal-loss']
assert all(load(name + '.json')['all_returned_buffers_blocked_read_and_finite'] for name in source_graphs)
assert c['source_replay_boundaries_exact'] and c['attribution_valid'] and o['attribution_valid'] and o['completed']
assert load('wrapper-outcomes.json') == {'probe_returncode': 0, 'replay_returncode': 0}
assert (p['process_count'], p['global_device_count'], p['local_device_count']) == (1, 4, 4)
assert all(device['kind'] == 'TPU v5' and device['process_index'] == 0 for device in p['devices'])
submission = load('submission-provenance.json')
for field, filename in [('diagnostic_sha256', 'native_jax_bf16_splash_15b_separate_capture.py'), ('wrapper_sha256', 'native_jax_block_replay_job_bootstrap.py'), ('frozen_helper_sha256', 'native_jax_bf16_attention_replay.py'), ('frozen_original_script_sha256', 'native_jax_bf16_splash_15b_block_replay.py')]:
    assert submission[field] == hashlib.sha256((root / filename).read_bytes()).hexdigest()
source_mapping = {'last_input': ('source-scan', 'last_input'), 'last_output': ('source-scan', 'last_output'), 'post_attention_residual': ('capture-without-duplicate-full', 'residual'), 'output_cotangent': ('capture-without-duplicate-full', 'output_cotangent'), 'tail_input_cotangent': ('capture-without-duplicate-full', 'tail_input_cotangent')}
for name, (graph, value) in source_mapping.items():
    assert load(graph + '.json')['results'][value]['sha256'] == p['source_operands'][name]['sha256']
for name in source_graphs:
    for record in load(name + '.json')['results'].values():
        assert len(record['addressable_shards']) == 4
        if record['shape']:
            assert record['shape'] == [8, 2048, 2048] and record['dtype'] == 'bfloat16'
            assert all(shard['shape'] == [4, 2048, 2048] for shard in record['addressable_shards'])
        else:
            assert record['dtype'] == 'float32'
report = {
    'verdict': 'Artifact integrity, source/operand boundary controls and standalone full/composed controls independently verified. Diagnostic-only localization; no full-Trainer acceptance.',
    'scope': 'Read-only retained archive/JSON/HLO hash/source audit, no jobs/source edits/downloads/numerical execution.',
    'source_commit': p['source_commit'], 'script_sha256': p['script_sha256'],
    'archive': {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': archive_sha, 'entries': len(members), 'all_archive_members_byte_identical_to_extraction': True, 'raw_bytes': sum(x.stat().st_size for x in r.rglob('*') if x.is_file()), 'script_manifest_entries': len(o['files']), 'script_files_with_manifest': o['files_with_manifest'], 'wrapper_files': 2},
    'hlo': {'files': len(hlo), 'all_complete_untruncated_hash_verified': True, 'bytes': o['hlo_bytes'], 'source_capture_bytes': o['source_capture_hlo_bytes'], 'vjp_bytes': o['vjp_hlo_bytes']},
    'source_file_sha_matches_git36e': source_matches,
    'parameter_count': p['parameter_count'], 'initial_parameter_sha256': p['parameter_sha256'], 'batch_sha256': p['batch_sha256'],
    'five_operand_hashes_match_original858f': frozen_operands,
    'all_capture_guards_exact': True, 'canonical_named_native_master_import_roundtrips_exact': True,
    'source_replay_boundaries_exact': True, 'all33_full_composed_outputs_exact': True,
    'all33_full_composed_result_hashes_independently_equal': full_composed_hash_matches,
    'all_returned_source_buffers_recorded_read_and_finite': True, 'all_recorded_numeric_statistics_finite': finite_record_numbers,
    'topology': {'processes': p['process_count'], 'global_devices': p['global_device_count'], 'local_devices': p['local_device_count'], 'devices': p['devices']},
    'stages': stages,
    'interpretation': [
        'Same captured legacy last-block x/r/y/ct_y/ct_r as original858f. Removing only duplicate full evaluation permits all strict source capture controls to pass; original combined-graph fault is preserved as a separate observation.',
        'Named/native prefix forward is exact. Prefix differences are inputVJP relativeL2 0.0588515%, inputnorm masterVJP0.565345%, Q masterVJP0.457671%; other active K/O/V masterVJPs are exact.',
        'Fixed-boundary tail has exact forward, inputVJP and all4active masterVJPs; remaining5parameterVJPs are unused zeros.',
        'Full and composed are exactly equal within each variant across11outputs. Named/native full forwards are exact; inputVJP and6masterVJPs differ (inputnorm,down,K,Q,V). Correction: six nonexact outputs include inputVJP plus five masterVJPs, as explicit stage mapping records.',
        'Full/composed standalone nonexact master locations differ from normal Trainer lastblock gradients (normal up/O differ; standalone down/K/V differ). This is a compiler-context-sensitive fragment diagnostic, not a faithful reproduction of every normal backward tensor.',
        'Every staged variant uses fixed named r/ct_r. Staged-versus-full differences conflate operand substitution,dtype and fusion; cannot call them pure fusion error. FP32-highest uses same Splash but is not exact arithmetic groundtruth.',
        'RelativeL2 denominator is the named/left fragment tensor norm; percentages are ratios multiplied by100. No changed tolerances or accepted model-level speed/gradient ratio follows from this diagnostic.',
    ],
    'limitations': p['limitations'],
}
# Derive exact counts to avoid prose ambiguity.
for stage, values in stages.items():
    values['nonexact_output_count'] = len(values['named_native_nonexact_outputs'])
    values['nonexact_master_vjp_count'] = sum(name not in ('forward', 'input_vjp') for name in values['named_native_nonexact_outputs'])
report['interpretation'][3] = 'Full and composed are exactly equal within each variant across11outputs. Named/native full forwards are exact; inputVJP and5masterVJPs differ (inputnorm,down,K,Q,V).'
path = root / 'native-jax-bf16-splash-15b-separate-vjp-packed-independent-audit.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(path, path.stat().st_size)
print(json.dumps({stage: {'exact': len(v['named_native_exact_outputs']), 'nonexact': list(v['named_native_nonexact_outputs'])} for stage, v in stages.items()}, indent=2))
