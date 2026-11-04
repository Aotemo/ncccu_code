"""Activate one explicitly frozen formal package once; never start GPU work."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
import uuid

_ACTIVE_PREP = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_training_topology(topology, positive_quota):
    require((topology['world_size'], topology['per_rank_batch'], topology['accumulation'],
             topology['reference_physical_batch'], topology['global_effective_batch'], topology['global_positive_quota'])
            == (8, 8, 1, 32, 64, positive_quota), 'formal eight-rank scientific topology differs')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def pinned(path, digest, root):
    from retrain.site_binding import evidence_path
    path = evidence_path(path, root)
    require(re.fullmatch(r'[0-9a-f]{64}', digest or '') and sha(path) == digest, 'evidence SHA differs: ' + str(path))
    return read(path)


def ident(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'start_ticks': fields[19],
                'argv': [part.decode('utf-8', 'replace') for part in (path / 'cmdline').read_bytes().split(b'\0') if part]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_alive(old, actual):
    return bool(actual and actual['state'] not in ('Z', 'X') and actual['start_ticks'] == str(old['start_ticks']))


def ensure_new_activation(prep, formal):
    for path in (Path(formal), Path(prep) / 'FORMAL-CLOCK.json', Path(prep) / 'ACTIVATION-FAILURE.json'):
        if path.exists() or path.is_symlink():
            raise FileExistsError('one-shot activation is already reserved: ' + str(path))


def verify_package(package, manifest, site, *, activation_script):
    require(manifest.get('schema_version') == 1 and manifest.get('git_commit') == site['git_commit']
            and manifest['formal_root'] == site['round_root'] and manifest['package_root'] == str(package)
            and manifest['activate_script_sha256'] == sha(activation_script), 'package deployment identity differs')
    actual = {}
    for path in package.rglob('*'):
        require(not path.is_symlink(), 'package symlink forbidden')
        if path.is_file():
            actual[str(path.relative_to(package))] = sha(path)
    require(actual == manifest['files'], 'package contains missing, changed or unmanifested files')
    require(not any(name.startswith(('control/', 'outputs/')) for name in actual), 'package contains previous execution state')
    reference = read(package / 'SOURCE-MANIFEST.reference.json')
    bundle = {str(path.relative_to(package / 'bundle')): sha(path) for path in (package / 'bundle').rglob('*.py')}
    digest = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(reference['after'] == bundle and reference['source_digest'] == digest == manifest['bundle_source_digest'],
            'package numerical source identity differs')
    require(read(package / 'SITE_BINDING.json') == site, 'package site binding differs')
    return digest


def verify_deployment(value, site, manifest, manifest_sha, package, prep):
    from retrain.site_binding import evidence_path
    require(value.get('schema_version') == 1 and value.get('status') == 'frozen_from_remote_commit'
            and value.get('scope') == 'formal_exp_v3' and value.get('git_commit') == site['git_commit']
            and value.get('remote_commit') == site['git_commit']
            and value.get('remote_url') == 'https://github.com/Aotemo/ncccu_code'
            and value.get('pod_uid') == site['pod']['uid']
            and value.get('package_manifest_sha256') == manifest_sha, 'remote Git/frozen deployment evidence differs')
    source = evidence_path(value['source_root'], site['work_root'])
    require(source.is_dir(), 'frozen repository source root missing')
    files = value['files']
    required = {'runtime/' + name: digest for name, digest in manifest['files'].items() if name.endswith('.py')}
    required.update({'tools/activation/' + name: sha(prep / name)
                     for name in ('activate.py', 'activate_audit.py', 'site_preflight.py')})
    for relative, digest in required.items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts and files.get(relative) == digest,
                'Git deployment omits a package source: ' + relative)
        target = evidence_path(source / path, source)
        require(target.is_file() and sha(target) == digest, 'actual frozen source SHA differs: ' + relative)
    return value


def verify_gpu_admission(value, site, *, manifest_sha, source_digest, cpu):
    from retrain.site_binding import verify_rank_devices
    require(value.get('schema_version') == 1 and value.get('status') == 'completed' and value.get('exit_code') == 0
            and value.get('git_commit') == site['git_commit'] and value.get('pod_uid') == site['pod']['uid']
            and value.get('package_manifest_sha256') == manifest_sha
            and value.get('bundle_source_digest') == source_digest
            and value.get('candidate') == site['science']['candidate'], 'GPU admission source/Pod/candidate differs')
    require(type(value.get('successful_updates')) is int and 0 <= value['successful_updates'] <= 2
            and type(value.get('wall_seconds')) in (int, float) and math.isfinite(value['wall_seconds'])
            and 0 < value['wall_seconds'] <= 300, 'GPU admission exceeded its bounded technical budget')
    for key in ('all_processes_exited', 'fit_only', 'fresh_initial_state_verified', 'fp32_collective_verified', 'finite_values_verified'):
        require(value.get(key) is True, 'GPU admission did not prove ' + key)
    require(value.get('live_processes') == [] and value.get('dev_evaluated') is False
            and value.get('official_evaluated') is False, 'GPU technical admission used forbidden data or remains live')
    require(value.get('initial_state_control') == cpu['initial_state_control'], 'GPU fresh identity differs from production CPU admission')
    proofs = verify_rank_devices(value['rank_proofs'], site)
    require(all(proof.get('CPU_only_software_proof') is False and proof.get('intraop_threads') == 1
                and proof.get('interop_threads') == 1 for proof in proofs),
            'GPU admission needs actual accelerator and exact rank thread proofs')
    processes = value['released_processes']
    require(isinstance(processes, list) and len(processes) >= len(site['gpus']) + 1
            and len({item['pid'] for item in processes}) == len(processes), 'GPU owner/rank exact exit identities required')
    for item in processes:
        require(type(item.get('pid')) is int and item['pid'] > 0 and str(item.get('start_ticks', '')).isdecimal()
                and isinstance(item.get('argv'), list) and item['argv'], 'GPU process identity is incomplete')
    require({proof.get('pid') for proof in proofs} <= {item['pid'] for item in processes},
            'GPU rank proofs lack exact exited process identities')
    return processes


def verify_preflight(value, site, package, *, now):
    require(value.get('schema_version') == 1 and value.get('exit_code') == 0
            and value.get('status') == 'site_identity_dependencies_and_declared_lease_checked'
            and value.get('Ready') is True and value.get('pod_uid') == site['pod']['uid']
            and value.get('round_id') == site['round_id']
            and value.get('site_binding_sha256') == sha(package / 'SITE_BINDING.json')
            and value.get('clock_policy_sha256') == sha(package / 'CLOCK_POLICY.json')
            and value.get('script_sha256') == sha(package.parent / 'site_preflight.py')
            and os.path.realpath(value['python']) == os.path.realpath(site['pod']['python'])
            and type(value.get('observed_at')) in (int, float) and math.isfinite(value['observed_at'])
            and 0 <= now - value['observed_at'] <= 120
            and value.get('CUDA_initialized') is False and value.get('GPU_started') is False
            and value.get('formal_T0_created') is False and value.get('training_lock_acquired') is False
            and value.get('lease_annotations') == {}, 'fresh CPU site preflight differs')
    total = read(package / 'CLOCK_POLICY.json')['total_seconds']
    require(value.get('required_seconds') == total and value['declared_lease_limits'], 'lease budget identity differs')
    for limit in value['declared_lease_limits']:
        if limit['deadline_at'] is not None:
            require(math.isfinite(limit['deadline_at']) and limit['deadline_at'] - now >= total,
                    'remaining declared lease cannot cover formal budget')
    return value


def verify_storage(work_root):
    """Keep the strict admission boundary and preserve the one actual reading."""
    free = shutil.disk_usage(work_root).free
    evidence = {'at': time.time(), 'work_root': str(work_root),
                'free_bytes': free, 'minimum_free_bytes': 20 * 1024**3}
    require(free > evidence['minimum_free_bytes'],
            f"insufficient storage: free_bytes={free} minimum_free_bytes={evidence['minimum_free_bytes']} work_root={work_root}")
    return evidence


def admit_now(package, site, processes):
    # Observe live processes/locks/cards; never acquire the training lock or signal work.
    sys.path.insert(0, str(package))
    import run_round as guard
    exact = []
    for old in processes:
        actual = ident(old['pid'])
        require(not process_alive(old, actual), 'previous exact GPU owner/rank remains live')
        exact.append({'expected': old, 'actual': actual, 'inactive': True})
    from retrain.site_binding import instance_lock
    lock = instance_lock(site); lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_size == 0, 'instance lock must be the actual empty regular lock')
        matching = []
        for line in Path('/proc/locks').read_text().splitlines():
            for word in line.split():
                parts = word.split(':')
                if len(parts) == 3:
                    try:
                        same = (int(parts[0], 16), int(parts[1], 16), int(parts[2])) == (
                            os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
                    except ValueError:
                        continue
                    if same:
                        matching.append(line)
        require(not matching, 'instance lock remains held')
    finally:
        os.close(descriptor)
    users = guard.device_users_before_start()
    require(not users, 'accelerator device handles remain occupied')
    resource = guard.runtime_safety_snapshot()
    require(len(resource['cards']) == len(site['gpus']) and all(
        card['busy_percent'] == 0 and card['vram_used_bytes'] < 64 * 1024**2 for card in resource['cards']),
        'every admitted device must be actually idle')
    storage = verify_storage(site['work_root'])
    return {'at': time.time(), 'exact_processes': exact, 'lock_released': True, 'lock_inode': info.st_ino,
            'lock_device': info.st_dev, 'lock_path': str(lock), 'proc_locks': matching,
            'resources': resource, 'device_users': users, 'lock_acquisition_attempted': False, 'signals_sent': False,
            'storage': storage}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--activate-formal', action='store_true', required=True)
    result.add_argument('--prep-root', type=Path, required=True)
    result.add_argument('--authorization-reference', required=True)
    result.add_argument('--site-binding', type=Path, required=True)
    result.add_argument('--site-binding-sha256', required=True)
    result.add_argument('--package-manifest-sha256', required=True)
    result.add_argument('--cpu-prep-receipt-sha256', required=True)
    for name in ('deployment-receipt', 'site-preflight'):
        result.add_argument('--' + name, type=Path, required=True)
        result.add_argument('--' + name + '-sha256', required=True)
    result.add_argument('--gpu-admission', type=Path)
    result.add_argument('--gpu-admission-sha256')
    result.add_argument('--pod-uid', required=True)
    result.add_argument('--ready', choices=['True'], required=True)
    result.add_argument('--observed-at', type=float, required=True)
    return result


def main(argv=None):
    global _ACTIVE_PREP
    _ACTIVE_PREP = None
    args = parser().parse_args(argv)
    prep = args.prep_root.resolve(strict=True); package = prep / 'package'
    sys.path.insert(0, str(package / 'bundle'))
    from retrain.site_binding import load_site, verify_live, require_cpu_channel, evidence_path, verify_gpu_cpu
    site = load_site(args.site_binding, expected_sha256=args.site_binding_sha256)
    require(site.get('scope') == 'formal_exp_v3' and site.get('source_state', {}).get('kind') != 'uncommitted_cpu_snapshot',
            'technical CPU snapshots cannot become formal activations')
    for path in (prep, args.site_binding, Path(site['round_root'])):
        evidence_path(path, site['work_root'])
    require(args.site_binding.resolve() == package / 'SITE_BINDING.json' and prep.is_relative_to(Path(site['work_root']) / 'technical-prep'),
            'activation requires the explicit isolated preparation package')
    require_cpu_channel(site)
    verify_live(site, pod_uid=args.pod_uid, ready=args.ready == 'True', observed_at=args.observed_at)
    require(bool(args.authorization_reference.strip()), 'actual authorization reference required')
    formal = Path(site['round_root'])
    ensure_new_activation(prep, formal)
    _ACTIVE_PREP = prep
    manifest_path = prep / 'PACKAGE-MANIFEST.json'
    manifest = pinned(manifest_path, args.package_manifest_sha256, site['work_root'])
    source_digest = verify_package(package, manifest, site, activation_script=__file__)
    preparation = pinned(prep / 'CPU-PREP-RECEIPT.json', args.cpu_prep_receipt_sha256, site['work_root'])
    require(preparation.get('status') == 'prepared_no_formal_clock_no_GPU' and preparation.get('exit_code') == 0
            and preparation.get('source_digest') == source_digest
            and preparation.get('package_manifest_sha256') == args.package_manifest_sha256, 'production CPU preparation differs')
    cpu = read(package / 'evidence/MODEL_CPU_ADMISSION.template.json')
    require(cpu.get('status') == 'completed' and cpu.get('formal_train_config_sha256') is None
            and cpu.get('gpu_initialized') is False and cpu.get('source_digest') == source_digest
            and cpu.get('checks') and all(item.get('passed') is True for item in cpu['checks'])
            and cpu.get('initial_state_control') == preparation['initial_state_control'], 'CPU candidate admission differs')
    require(preparation.get('cpu_receipt_sha256') == cpu['original_receipt_sha256'], 'CPU preparation pins another model receipt')
    original = pinned(cpu['original_receipt_path'], cpu['original_receipt_sha256'], site['work_root'])
    require(original.get('exit_code') == 0 and original.get('source_digest') == source_digest
            and original.get('initial_state_control') == cpu['initial_state_control'], 'actual CPU model receipt differs')
    deployment = pinned(args.deployment_receipt, args.deployment_receipt_sha256, site['work_root'])
    verify_deployment(deployment, site, manifest, args.package_manifest_sha256, package, prep)
    if args.gpu_admission is not None:
        gpu = pinned(args.gpu_admission, args.gpu_admission_sha256, site['work_root'])
        processes = verify_gpu_admission(gpu, site, manifest_sha=args.package_manifest_sha256, source_digest=source_digest, cpu=cpu)
    else:
        require(args.gpu_admission_sha256 is None, 'GPU admission SHA requires an explicit receipt')
        gpu = {'status': 'not_run_for_lr_only', 'GPU_started': False,
               'actual_training_initial_gate_required': True}
        processes = []
    preflight = pinned(args.site_preflight, args.site_preflight_sha256, site['work_root'])
    verify_preflight(preflight, site, package, now=time.time())
    verify_gpu_cpu(site)  # Validate the future GPU allocation without changing this CPU channel's mask.
    os.environ.update(NCCCU_SITE_BINDING=str(package / 'SITE_BINDING.json'),
                      NCCCU_SITE_BINDING_SHA256=args.site_binding_sha256)
    handoff = admit_now(package, site, processes)
    policy = read(package / 'CLOCK_POLICY.json')
    fields = ('total_seconds', 'binding_seconds', 'train_hard_seconds', 'freeze_hard_seconds', 'official_hard_seconds', 'cleanup_seconds')
    require(policy['round_id'] == site['round_id'] and policy.get('clock_reset_allowed') is False
            and all(type(policy[key]) is int and policy[key] > 0 for key in fields)
            and policy['binding_seconds'] + policy['train_hard_seconds'] + policy['freeze_hard_seconds']
                + policy['official_hard_seconds'] + policy['cleanup_seconds'] <= policy['total_seconds'],
            'frozen round clock policy differs')
    sys.path.insert(0, str(package / 'bundle'))
    from retrain.config import load_config, job_contract
    from retrain.execution_policy import resolve_topology
    from retrain.fresh import validate_fresh
    from retrain.experiment91 import binding, schedule, science_configuration
    cfg = load_config(package / 'TRAIN_CONFIG.json', environ={})
    job, topology, execution_policy = resolve_topology(cfg, available_accelerators=len(site['gpus']))
    require(cfg['mode'] == 'train' and cfg['resume'] is False and len(cfg['jobs']) == 1
            and cfg['execution']['devices'] == ['cuda:' + str(rank) for rank in range(len(site['gpus']))]
            and cfg['output_root'] == str(formal / 'outputs') and cfg['data']['root'] == site['data_root']
            and cfg['data']['identity_path'] == site['train_identity'], 'formal config paths/mode/devices differ from site')
    fresh = read(package / 'FRESH-SPEC.template.json')
    topology['stop_after_updates'] = schedule()['stop_after_updates']
    verify_training_topology(topology, science_configuration()['job']['positive_per_batch'])
    require(fresh['execution_policy'] == execution_policy and fresh['source_digest'] == source_digest
            and cpu['job_contract'] == job_contract(job) and cpu['topology'] == topology, 'CPU/fresh runtime contract differs')
    fresh['round_deadline_at'] = time.time() + policy['total_seconds']
    identity = {'format': 'glaucoma-retrain-v1', 'source_digest': fresh['source_digest'], 'manifest_digest': binding()['manifest_digest'],
        'job': job_contract(job), 'initialization': fresh['initialization'], 'phase_parent_sha256': None,
        'distributed': topology, 'fresh': fresh, 'experiment91': binding()}
    validate_fresh(fresh, identity, job, topology, schedule()['steps_per_cycle'], mode='train')
    verify_live(site, pod_uid=args.pod_uid, ready=True, observed_at=args.observed_at)
    verify_preflight(preflight, site, package, now=time.time())
    # First formal mutation. Failure after this point must preserve the same clock/root.
    t0 = time.time()
    clock = {'round_id': site['round_id'], 'token': uuid.uuid4().hex, 'origin': 'explicit formal activation',
        'authorization_reference': args.authorization_reference, 'first_implementation_at': t0,
        'implementation_deadline_at': t0 + policy['binding_seconds'], 'deadline_at': t0 + policy['total_seconds'],
        'train_budget_seconds': policy['train_hard_seconds'], 'clock_reset_allowed': False}
    write(prep / 'FORMAL-CLOCK.json', clock)
    formal.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, formal)
    for name in ('control', 'evidence', 'outputs'):
        (formal / name).mkdir(exist_ok=True)
    for rel, digest in manifest['files'].items():
        require(sha(formal / rel) == digest, 'formal copied package differs: ' + rel)
    os.environ.update(NCCCU_SITE_BINDING=str(formal / 'SITE_BINDING.json'))
    write(formal / 'control/round-clock.json', clock)
    write(formal / 'control/predecessor-handoff.json', handoff)
    for name, value in (('DEPLOYMENT.json', deployment), ('GPU_ADMISSION.json', gpu), ('SITE_PREFLIGHT.json', preflight)):
        write(formal / 'evidence' / name, value)
    fresh['round_deadline_at'] = clock['deadline_at']
    identity = {'format': 'glaucoma-retrain-v1', 'source_digest': fresh['source_digest'], 'manifest_digest': binding()['manifest_digest'],
        'job': job_contract(job), 'initialization': fresh['initialization'], 'phase_parent_sha256': None,
        'distributed': topology, 'fresh': fresh, 'experiment91': binding()}
    validate_fresh(fresh, identity, job, topology, schedule()['steps_per_cycle'], mode='train')
    write(formal / 'control/train-fresh-spec.json', fresh)
    cpu['formal_train_config_sha256'] = sha(formal / 'TRAIN_CONFIG.json')
    write(formal / 'evidence/MODEL_CPU_ADMISSION.json', cpu)
    write(formal / 'FREEZE.json', {'round_id': clock['round_id'], 'science_frozen': True, 'experiment91': binding(),
        'files': {name: sha(formal / name) for name in ('PLAN.md', 'TRAIN_CONFIG.json', 'ASSETS.json', 'CLOCK_POLICY.json', 'SITE_BINDING.json')},
        'source_digest': source_digest, 'git_commit': site['git_commit'], 'at': time.time()})
    source_files = {str(path.relative_to(formal)): sha(path) for path in sorted(formal.rglob('*')) if path.is_file()}
    runtime = {'round_id': clock['round_id'], 'at': time.time(), 'implementation_ready': True,
        'git_commit': site['git_commit'], 'site_binding_sha256': sha(formal / 'SITE_BINDING.json'),
        'source_digest': source_digest, 'bundle_source_files': read(formal / 'SOURCE-MANIFEST.reference.json')['after'],
        'source_files': source_files, 'public_weight_sha256': job['initialization']['sha256'],
        'clock_sha256': sha(formal / 'control/round-clock.json'), 'freeze_sha256': sha(formal / 'FREEZE.json'),
        'effective_output_root': str(formal / 'outputs'), 'initial_state_control': cpu['initial_state_control'],
        'fresh_spec_sha256': {'train': sha(formal / 'control/train-fresh-spec.json')},
        'package_manifest_sha256': args.package_manifest_sha256, 'cpu_prep_receipt_sha256': args.cpu_prep_receipt_sha256,
        'deployment_receipt_sha256': args.deployment_receipt_sha256, 'gpu_admission_sha256': args.gpu_admission_sha256,
        'gpu_admission_status': gpu['status']}
    write(formal / 'RUNTIME_BINDING.json', runtime)
    write(formal / 'control/binding-receipt.json', {'runtime_binding_sha256': sha(formal / 'RUNTIME_BINDING.json'), 'at': time.time()})
    require(time.time() < clock['implementation_deadline_at'], 'formal binding window elapsed')
    result = {'status': 'formally_bound_not_dispatched', 'exit_code': 0, 'root': str(formal), 'clock': clock,
        'binding_sha256': sha(formal / 'RUNTIME_BINDING.json'), 'GPU_started': False, 'source_digest': source_digest,
        'git_commit': site['git_commit'], 'site_binding_sha256': args.site_binding_sha256}
    write(formal / 'evidence/FORMAL_BINDING_RECEIPT.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if _ACTIVE_PREP is not None:
            failure = _ACTIVE_PREP / 'ACTIVATION-FAILURE.json'
            if not failure.exists() and not failure.is_symlink():
                clock = _ACTIVE_PREP / 'FORMAL-CLOCK.json'
                write(failure, {'status': 'failed_no_reset', 'error': repr(error),
                    'formal_clock': read(clock) if clock.is_file() else None})
        raise
