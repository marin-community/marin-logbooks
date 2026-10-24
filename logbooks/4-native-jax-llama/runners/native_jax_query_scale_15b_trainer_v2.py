"""Run matched Trainer cases with bounded diagnostics; no submission logic."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

MAX_EXEC_BYTES = 96 * 1024

BENCHMARK_CHILD_BOOTSTRAP = 'import hashlib,importlib,importlib.util,json,pathlib,runpy,sys\nproof=json.loads(sys.argv[1])\nbenchmark=pathlib.Path(sys.argv[2]).resolve()\nroot=pathlib.Path(proof["worktree"]).resolve()\nactual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in proof["source_file_sha256"]}\nif actual!=proof["source_file_sha256"]:raise ValueError("Actual benchmark child source bytes differ")\nspec=importlib.util.spec_from_file_location("native_jax_benchmark_import_proof",benchmark)\nmodule=importlib.util.module_from_spec(spec)\nsys.modules[spec.name]=module\nspec.loader.exec_module(module)\ncore=importlib.import_module("levanter.grug.attention._core")\nlevanter=importlib.import_module("levanter")\nhaliax=importlib.import_module("haliax")\npaths={"core":str(pathlib.Path(core.__file__).resolve()),"levanter":str(pathlib.Path(levanter.__file__).resolve()),"haliax":str(pathlib.Path(haliax.__file__).resolve()),"benchmark":str(pathlib.Path(module.__file__).resolve())}\nexpected_paths={"core":root/"lib/levanter/src/levanter/grug/attention/_core.py","levanter":root/"lib/levanter/src/levanter/__init__.py","haliax":root/"lib/haliax/src/haliax/__init__.py","benchmark":benchmark}\nif any(pathlib.Path(paths[key])!=expected_paths[key] for key in paths):raise ValueError("Benchmark child imported a module from another tree")\nbarrier="optimization_barrier" in core._tpu_splash_attention.__code__.co_names\nif not barrier:raise ValueError("Imported candidate function lacks query-scale barrier")\nrecord={"verified":True,"source_pin_declared":proof["source_commit"],"source_tree_clean_is_declared_not_git_checked":True,"actual_source_file_sha256":actual,"modules":paths,"actual_function_contains_barrier":barrier,"core_sha256":hashlib.sha256(pathlib.Path(core.__file__).read_bytes()).hexdigest(),"dont_write_bytecode":sys.dont_write_bytecode,"pycache_prefix":sys.pycache_prefix,"execution":"Import provenance and runpy.run_path of unchanged benchmark execute in this same child process"}\noutput=pathlib.Path(proof["record_path"])\noutput.parent.mkdir(parents=True,exist_ok=True)\noutput.write_text(json.dumps(record,indent=2,sort_keys=True)+"\\n")\nsys.argv=[str(benchmark),*sys.argv[3:]]\nrunpy.run_path(str(benchmark),run_name="__main__")\n'


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def source_guard(config):
    actual = {name: sha256(Path(name)) for name in config['source_file_sha256']}
    if actual != config['source_file_sha256']:
        raise ValueError('Actual source files differ from the pinned candidate')
    return actual


def archive_diagnostics(output, archive, limits):
    files, skipped = [], []
    for path in sorted(output.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink in diagnostic output: {path}')
        if not path.is_file():
            continue
        relative = str(path.relative_to(output))
        if 'perfetto' in path.name or '.trace.json' in path.name:
            skipped.append(relative)
            continue
        if not (path.suffix == '.json' or path.name.endswith('.xplane.pb') or path.name.endswith('hlo.txt')):
            raise ValueError(f'Unexpected diagnostic output: {relative}')
        if path.stat().st_size > limits['per_file_bytes']:
            raise ValueError(f'Diagnostic file exceeds hard cap; not truncated: {relative} ({path.stat().st_size})')
        files.append(path)
    uncompressed = sum(path.stat().st_size for path in files)
    print('DIAGNOSTIC_PREFLIGHT ' + json.dumps({'files': len(files), 'uncompressed_bytes': uncompressed,
        'largest_file_bytes': max((p.stat().st_size for p in files), default=0)}, sort_keys=True), flush=True)
    if len(files) > limits['files'] or uncompressed > limits['uncompressed_bytes']:
        raise ValueError('Diagnostic archive exceeds hard count/raw cap; not truncated')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(output)), recursive=False)
    compressed = archive.stat().st_size
    if compressed > limits['compressed_bytes']:
        raise ValueError('Compressed diagnostic cap exceeded')
    return {'compressed_bytes': compressed, 'uncompressed_bytes': uncompressed, 'entries': len(files),
        'sha256': sha256(archive), 'skipped_perfetto': skipped,
        'files': [{'path': str(p.relative_to(output)), 'bytes': p.stat().st_size} for p in files]}


def matched_report_guards(report, config, case):
    provenance = report['provenance']
    expected = dict(config['expected_benchmark_config'], compute_dtype=case['name'], profile_steps=case['profile_steps'])
    checks = {
        'benchmark_config_exact': provenance['config'] == expected,
        'source_pin_exact': provenance['git_commit'] == config['source_commit'],
        'source_clean_declared': provenance['source_tree_state'] == 'clean',
        'initial_within_pair_exact': report['initial_parameter_parity']['allclose'] and report['initial_parameter_parity']['max_abs'] == 0,
        'parameter_and_moments_float32': provenance['parameter_dtype'] == 'float32' and provenance['optimizer_moment_dtype'] == 'float32',
        'matmul_precision_exact': provenance['jax_default_matmul_precision'] == case['matmul_precision'],
        'no_hook_first_state_checks_present': all(name in report['parity'] and name in report['acceptance'] for name in ['execution_parameters_after_one_step','execution_adam_mu','execution_adam_nu','execution_adam_count']),
        'all_original_manifest_guards_completed': all(name in report['variants'][variant]['layouts'] for variant in ['legacy','native'] for name in ['parameters','gradients','updates','parameters_after_one_step','adam_mu','adam_nu','adam_count','execution_parameters_after_one_step','execution_adam_mu','execution_adam_nu','execution_adam_count','activations']),
        'all_measured_losses_present': all(len(report['variants'][variant]['measured_step_losses']) == expected['measured_steps'] for variant in ['legacy','native'])}
    if config['historical_hash_guards_applicable']:
        checks.update(parameter_count_exact=provenance['parameter_count'] == config['parameter_count'],
            historical_initial_parameter_hash_exact=provenance['parameter_sha256'] == config['expected_parameter_sha256'],
            native_import_parameter_hash_exact=provenance['native_parameter_sha256'] == config['expected_parameter_sha256'],
            historical_batch_hash_exact=provenance['batch_sha256'] == config['expected_batch_sha256'])
    return {'passed': all(checks.values()), 'checks': checks, 'historical_hash_guards_applicable': config['historical_hash_guards_applicable'],
        'numerical_comparison_valid': report['comparison_valid'], 'numerical_criteria_changed': False,
        'comparison_scope': 'Matched full Trainer step and independent trajectory. Raw physical placements are retained; the unchanged benchmark hard-asserts all canonical manifests before producing this report.'}


def main():
    config = json.loads(sys.argv[1])
    if 'jax' in sys.modules:
        raise RuntimeError('Runner parent must not initialize JAX')
    if sha256(Path(__file__)) != config['runner_sha256']:
        raise ValueError('Runner bytes differ from the reviewed preparation')
    if os.environ.get('XLA_FLAGS') != '--xla_allow_excess_precision=false':
        raise ValueError('Exact excess-precision flag required')
    actual_source = source_guard(config)
    output = Path(os.environ['IRIS_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a fresh empty output directory')
    probe_script = "import jax,json;print(json.dumps({'backend':jax.default_backend(),'process_count':jax.process_count(),'global_device_count':jax.device_count(),'local_device_count':jax.local_device_count(),'devices':[{'id':d.id,'kind':d.device_kind,'process_index':d.process_index,'description':str(d)} for d in jax.local_devices()]}))"
    probe_process = subprocess.run([sys.executable, '-c', probe_script], text=True, capture_output=True, check=True)
    print(probe_process.stderr, end='', file=sys.stderr)
    probe = json.loads(probe_process.stdout)
    if probe['backend'] != config['expected_backend'] or probe['process_count'] != 1 or probe['global_device_count'] != 4 or probe['local_device_count'] != 4:
        raise ValueError('Expected exactly one process/four global and local devices')
    if config['expected_backend'] == 'tpu' and not all('v5' in device['kind'].lower() and 'lite' not in device['kind'].lower() for device in probe['devices']):
        raise ValueError('Expected v5p devices')
    (output / 'device-probe.json').write_text(json.dumps(probe, indent=2) + '\n')
    (output / 'submission-provenance.json').write_text(json.dumps(config | {'actual_source_file_sha256': actual_source, 'parent_jax_imported': False}, indent=2) + '\n')
    print('DEVICE_PROBE ' + json.dumps(probe, sort_keys=True), flush=True)
    outcomes, guards = [], {}
    for case in config['cases']:
        environment = dict(os.environ, IRIS_OUTPUT_DIR=str(output / case['name']), JAX_DEFAULT_MATMUL_PRECISION=case['matmul_precision'],
            PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(Path('lib/levanter/src').resolve()) + ':' + str(Path('lib/haliax/src').resolve()))
        environment['PYTHONPYCACHEPREFIX'] = tempfile.mkdtemp(prefix='native-jax-trainer-import-' + case['name'] + '-')
        runtime_path = output / case['name'] / 'runtime-imports.json'
        runtime_proof = {'worktree': str(Path.cwd()), 'source_commit': config['source_commit'],
            'source_file_sha256': config['source_file_sha256'], 'record_path': str(runtime_path)}
        command = [sys.executable, '-c', BENCHMARK_CHILD_BOOTSTRAP, json.dumps(runtime_proof, separators=(',', ':')),
            'lib/levanter/scripts/bench/bench_llama_native.py', '--mode', 'compare', '--attention-backend', config['attention_backend'], *config['common_args'],
            '--compute-dtype', case['name'], '--profile-steps', str(case['profile_steps']), '--source-commit', config['source_commit'], '--source-tree-state', 'clean',
            '--source-bundle-marker', config['job_name'], '--dump-hlo']
        argv_bytes = sum(len(value.encode()) + 1 for value in command)
        env_bytes = sum(len((key + '=' + value).encode()) + 1 for key, value in environment.items())
        if argv_bytes + env_bytes >= MAX_EXEC_BYTES:
            raise ValueError('Child command/environment exceeds96KiB')
        source_guard(config)
        result = subprocess.run(command, env=environment)
        report_path = output / case['name'] / 'compare.json'
        if report_path.is_file():
            guards[case['name']] = matched_report_guards(json.loads(report_path.read_text()), config, case)
        else:
            guards[case['name']] = {'passed': False, 'reason': 'Benchmark did not produce compare.json'}
        runtime_verified = runtime_path.is_file() and json.loads(runtime_path.read_text())['verified']
        guards[case['name']]['runtime_child_imports_verified'] = runtime_verified
        guards[case['name']]['passed'] = guards[case['name']]['passed'] and runtime_verified
        outcomes.append({'case': case['name'], 'returncode': result.returncode, 'matched_guards_passed': guards[case['name']]['passed'],
            'argv_bytes': argv_bytes, 'environment_bytes': env_bytes})
        (output / 'case-outcomes.json').write_text(json.dumps(outcomes, indent=2) + '\n')
        (output / 'matched-guards.json').write_text(json.dumps(guards, indent=2) + '\n')
    source_guard(config)
    archive = output.parent / (output.name + '-diagnostics.tar.gz')
    summary = archive_diagnostics(output, archive, config['archive_limits'])
    print('DIAGNOSTIC_BACKUP ' + json.dumps(summary | {'uri': config['backup_uri'], 'cases': outcomes}, sort_keys=True), flush=True)
    backup = subprocess.run([sys.executable, '-m', 'rigging.fsutil.cli', 'cp', '-n', str(archive), config['backup_uri']])
    print('DIAGNOSTIC_BACKUP_RESULT ' + json.dumps({'returncode': backup.returncode, 'uri': config['backup_uri']}, sort_keys=True), flush=True)
    raise SystemExit(1 if backup.returncode or any(case['returncode'] or not case['matched_guards_passed'] for case in outcomes) else 0)


if __name__ == '__main__':
    main()
