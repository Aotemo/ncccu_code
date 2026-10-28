"""Future formal activation only; CPU prepare never invokes main(). No GPU spawn."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
UID = 'eec33bd6-b567-40ef-b866-633e272ea019'
BASE = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027')
F = BASE / 'rounds/R91-R50-IMAGENETV2-9010'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def ident(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'start_ticks': fields[19],
                'argv': [a.decode() for a in (path / 'cmdline').read_bytes().split(b'\0') if a]}
    except FileNotFoundError:
        return None


def process_alive(old, actual):
    return bool(actual and actual['state'] not in ('Z', 'X') and actual['start_ticks'] == str(old['start_ticks']))


def admit_now(package):
    # These live reads intentionally occur only at formal activation, never prep.
    sys.path.insert(0, str(package))
    import run_round as guard
    previous = BASE / 'rounds/N15-R50-RGB01-SPATIAL'
    owner = read(previous / 'control/owner.json')['owner']
    component = read(previous / 'control/train-component-launch.json')['gate_identity']
    config = read(previous / 'TRAIN_CONFIG.json')
    ranks = [read(p) for p in sorted((previous / 'outputs' / config['run_name'] / 'rank-control').glob('rank-*-launch.json'))]
    expected = [owner, component, *ranks]
    assert len(expected) == 6
    processes = []
    for old in expected:
        actual = ident(old['pid'])
        inactive = not process_alive(old, actual)
        assert inactive, (old, actual)
        processes.append({'expected': old, 'actual': actual, 'inactive': inactive})
    lock = BASE.parent / ('.instance-owner-' + UID + '.lock')
    assert lock.stat().st_ino == 18030447997
    matching = [l for l in Path('/proc/locks').read_text().splitlines() if len(l.split()) > 5 and l.split()[5].endswith(':' + str(lock.stat().st_ino))]
    assert not matching
    assert not guard.device_users_before_start()
    resource = guard.runtime_safety_snapshot()
    assert len(resource['cards']) == 4 and all(c['busy_percent'] == 0 and c['vram_used_bytes'] < 64 * 1024**2 for c in resource['cards'])
    assert shutil.disk_usage(BASE).free > 20 * 1024**3
    return {'at': time.time(), 'exact_processes': processes, 'lock_released': True, 'lock_inode': lock.stat().st_ino,
            'proc_locks': matching, 'resources': resource, 'lock_acquisition_attempted': False, 'signals_sent': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate-formal', action='store_true', required=True)
    parser.add_argument('--authorization-reference', required=True)
    parser.add_argument('--pod-uid', required=True)
    parser.add_argument('--ready', required=True)
    args = parser.parse_args()
    assert args.authorization_reference.strip() and args.pod_uid == UID and args.ready == 'True'
    assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
    assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
    package = HERE / 'package'
    prep = read(HERE / 'STAGE3-PREP-RECEIPT.json')
    assert prep['status'] == 'prepared_no_formal_clock_no_GPU' and prep['exit_code'] == 0
    package_manifest = read(HERE / 'PACKAGE-MANIFEST.json')
    for rel, digest in package_manifest['files'].items():
        assert sha(package / rel) == digest, rel
    assert not F.exists() and not F.is_symlink() and not (HERE / 'FORMAL-CLOCK.json').exists(), 'No clock/root reset or second activation'
    handoff = admit_now(package)
    policy = read(package / 'CLOCK_POLICY.json')
    # First formal mutation: preserve this clock even if following binding fails.
    t0 = time.time()
    clock = {'round_id': policy['round_id'], 'token': uuid.uuid4().hex, 'origin': 'A routed R91 formal activation',
        'authorization_reference': args.authorization_reference, 'first_implementation_at': t0,
        'implementation_deadline_at': t0 + policy['binding_seconds'], 'deadline_at': t0 + policy['total_seconds'],
        'train_budget_seconds': policy['train_hard_seconds'], 'clock_reset_allowed': False}
    write(HERE / 'FORMAL-CLOCK.json', clock)
    shutil.copytree(package, F)
    for name in ('control', 'evidence', 'outputs'):
        (F / name).mkdir(exist_ok=True)
    for rel, digest in package_manifest['files'].items():
        assert sha(F / rel) == digest, rel
    write(F / 'control/round-clock.json', clock)
    write(F / 'control/predecessor-handoff.json', handoff)
    fresh = read(F / 'FRESH-SPEC.template.json')
    fresh['round_deadline_at'] = clock['deadline_at']
    write(F / 'control/train-fresh-spec.json', fresh)
    cpu = read(F / 'evidence/MODEL_CPU_ADMISSION.template.json')
    cpu['formal_train_config_sha256'] = sha(F / 'TRAIN_CONFIG.json')
    write(F / 'evidence/MODEL_CPU_ADMISSION.json', cpu)
    sys.path[:0] = [str(F / 'bundle')]
    from retrain.config import load_config, job_contract
    from retrain.execution_policy import resolve_topology
    from retrain.fresh import validate_fresh
    from retrain.experiment91 import binding, schedule
    cfg = load_config(F / 'TRAIN_CONFIG.json', environ={})
    job, topology, execution_policy = resolve_topology(cfg, available_accelerators=4)
    topology['stop_after_updates'] = schedule()['stop_after_updates']
    identity = {'format': 'glaucoma-retrain-v1', 'source_digest': fresh['source_digest'], 'manifest_digest': binding()['manifest_digest'],
        'job': job_contract(job), 'initialization': fresh['initialization'], 'phase_parent_sha256': None,
        'distributed': topology, 'fresh': fresh, 'experiment91': binding()}
    validate_fresh(fresh, identity, job, topology, schedule()['steps_per_cycle'], mode='train')
    write(F / 'FREEZE.json', {'round_id': clock['round_id'], 'science_frozen': True, 'experiment91': binding(),
        'files': {name: sha(F / name) for name in ('PLAN.md', 'TRAIN_CONFIG.json', 'ASSETS.json', 'CLOCK_POLICY.json')},
        'source_digest': fresh['source_digest'], 'at': time.time()})
    source_files = {str(p.relative_to(F)): sha(p) for p in sorted(F.rglob('*')) if p.is_file()}
    tested = read(F / 'SOURCE-MANIFEST.reference.json')['after']
    runtime = {'round_id': clock['round_id'], 'at': time.time(), 'implementation_ready': True,
        'source_digest': fresh['source_digest'], 'bundle_source_files': tested, 'source_files': source_files,
        'public_weight_sha256': job['initialization']['sha256'], 'clock_sha256': sha(F / 'control/round-clock.json'),
        'freeze_sha256': sha(F / 'FREEZE.json'), 'effective_output_root': str(F / 'outputs'),
        'initial_state_control': cpu['initial_state_control'], 'fresh_spec_sha256': {'train': sha(F / 'control/train-fresh-spec.json')}}
    write(F / 'RUNTIME_BINDING.json', runtime)
    write(F / 'control/binding-receipt.json', {'runtime_binding_sha256': sha(F / 'RUNTIME_BINDING.json'), 'at': time.time()})
    assert time.time() < clock['implementation_deadline_at']
    result = {'status': 'formally_bound_not_dispatched', 'exit_code': 0, 'root': str(F), 'clock': clock,
        'binding_sha256': sha(F / 'RUNTIME_BINDING.json'), 'GPU_started': False, 'source_digest': fresh['source_digest']}
    write(F / 'evidence/FORMAL_BINDING_RECEIPT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        path = HERE / 'ACTIVATION-FAILURE.json'
        if not path.exists():
            write(path, {'status': 'failed_no_reset', 'error': repr(error), 'formal_clock': read(HERE / 'FORMAL-CLOCK.json') if (HERE / 'FORMAL-CLOCK.json').exists() else None})
        raise
