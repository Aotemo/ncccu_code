"""Read-only compute-Pod CPU admission; does not touch the instance lock."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / 'preflight'
BASE = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2')
UID = 'eec33bd6-b567-40ef-b866-633e272ea019'
POD = 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
N06 = BASE / 'auroc999-20261027/rounds/N06'
R50 = BASE / 'auroc999-20261027/technical-prep/B-server-cpu-1793122900/reference-source/adapter'
N15 = BASE / 'auroc999-20261027/rounds/N15-R50-RGB01-SPATIAL'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with (OUT / name).open('xb') as stream:
        stream.write(raw)
    return {'path': str(OUT / name), 'sha256': sha(raw), 'bytes': len(raw)}


def process(pid):
    root = Path('/proc') / str(pid)
    try:
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'state': stat[0], 'start_ticks': stat[19],
                'argv': [x.decode(errors='replace') for x in (root / 'cmdline').read_bytes().split(b'\0') if x]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    assert sys.executable == '/usr/local/python3.12/bin/python3.12'
    assert sys.version_info[:2] == (3, 12)
    assert socket.gethostname() == POD
    assert (HERE / 'POD-IDENTITY.txt').read_text().splitlines() == [UID, 'True']
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
    assert all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    import numpy
    import torch
    import PIL
    import torchvision
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert torch.get_num_threads() <= 2 and torch.get_num_interop_threads() <= 2
    env = {'at': time.time(), 'pod_uid': UID, 'ready': True, 'hostname': socket.gethostname(),
           'python': sys.executable, 'python_version': sys.version, 'nice': os.getpriority(os.PRIO_PROCESS, 0),
           'dependencies': {'numpy': numpy.__version__, 'torch': torch.__version__, 'Pillow': PIL.__version__, 'torchvision': torchvision.__version__},
           'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
           'hidden_devices': {k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
           'cuda_initialized_before': False, 'cuda_initialized_after': torch.cuda.is_initialized(), 'GPU_enumeration_called': False}
    write('ENVIRONMENT.json', env)
    deployed = []
    for name in ('preflight.py', 'generate_split.py', 'verify_stage1.py', 'deploy_and_run.sh', 'README.md'):
        path = HERE / name
        assert path.is_file() and not path.is_symlink(), str(path)
        raw = path.read_bytes()
        deployed.append({'relative_path': name, 'path': str(path), 'bytes': len(raw), 'sha256': sha(raw)})
    with (HERE / 'DEPLOYMENT.json').open('x') as stream:
        json.dump({'status': 'DEPLOYED_SOURCE_MANIFEST', 'pod_uid': UID, 'root': str(HERE), 'files': deployed}, stream, indent=2)
        stream.write('\n')
    (HERE / 'DEPLOYMENT.json').chmod(0o444)
    lock = BASE / ('.instance-owner-' + UID + '.lock')
    locks = Path('/proc/locks').read_text().splitlines()
    lock_observation = {'path': str(lock), 'observed_at': time.time(), 'exists': lock.exists(),
                        'acquisition_attempted': False, 'lock_file_opened': False, 'signals_sent': False}
    if lock.exists():
        st = lock.stat()
        matching = []
        for line in locks:
            fields = line.split()
            for field in fields:
                parts = field.split(':')
                if len(parts) == 3:
                    try:
                        if (int(parts[0], 16), int(parts[1], 16), int(parts[2])) == (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
                            matching.append(line)
                    except ValueError:
                        pass
        lock_observation.update(inode=st.st_ino, device_major=os.major(st.st_dev), device_minor=os.minor(st.st_dev), proc_lock_records=matching)
    observed = []
    for root in sorted(Path('/proc').iterdir(), key=lambda p: p.name):
        if root.name.isdigit():
            record = process(root.name)
            if record and record['argv'] and any(term in ' '.join(record['argv']) for term in ('python', 'torchrun', 'run_round.py', 'train.py', 'target_owner', 'round_owner')):
                observed.append(record)
    records = []
    snapshots = OUT / 'reference-text'
    snapshots.mkdir()
    sources = [N06 / 'bundle/retrain' / n for n in ('data.py', 'age.py', 'persistence.py', 'engine.py')]
    sources += [R50 / 'bundle/retrain' / n for n in ('imagenet_resnet.py', 'models.py', 'ema.py', 'engine.py', 'n05_scoring.py')]
    sources += [N15 / 'control' / n for n in ('owner.json', 'dispatch.json', 'done.json', 'round-clock.json', 'resource-admission.json')]
    sources += [N15 / n for n in ('TRAIN_CONFIG.json', 'CLOCK_POLICY.json', 'owner.py', 'run_round.py')]
    for number, path in enumerate(sources):
        record = {'path': str(path), 'exists': path.exists()}
        if path.exists():
            assert path.is_file() and not path.is_symlink(), str(path)
            assert path.stat().st_size < 2_000_000, str(path)
            raw = path.read_bytes()
            saved = snapshots / ('%02d-' % number + path.name)
            saved.write_bytes(raw)
            record.update(sha256=sha(raw), bytes=len(raw), copied_to=str(saved))
            if path.parent == N15 / 'control' and path.name in ('owner.json', 'done.json'):
                value = json.loads(raw)
                record['value'] = value
                owner = value.get('owner')
                if isinstance(owner, dict) and isinstance(owner.get('pid'), int):
                    record['current_pid_observation'] = process(owner['pid'])
        records.append(record)
    assert all(x['exists'] for x in records[:7]), 'Required native metadata/R50 adapter source absent'
    source_manifest = write('SOURCE-INVENTORY.json', records)
    write('OWNER-LOCK-PROCESSES.json', {'at': time.time(), 'lock': lock_observation, 'processes': observed,
        'source_inventory': source_manifest, 'gpu_training_launch_attempted': False, 'signals_sent': False,
        'warning': 'Read-only snapshot; does not certify future exclusive GPU admission.'})
    mem = {p[0].rstrip(':'): int(p[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(p := line.split()) >= 2}
    disk = os.statvfs(BASE)
    write('CPU-RESOURCE-SNAPSHOT.json', {'at': time.time(), 'meminfo_kib': mem,
        'filesystem_available_bytes': disk.f_bavail * disk.f_frsize, 'VRAM_enumerated': False,
        'formal_GPU_budget_or_admission_created': False})
    assert not torch.cuda.is_initialized()
    return {'status': 'passed', 'exit_code': 0, 'environment': env, 'deployment_manifest': str(HERE / 'DEPLOYMENT.json'),
            'source_inventory': source_manifest, 'lock_acquisition_attempted': False, 'GPU_started': False}


if __name__ == '__main__':
    OUT.mkdir()
    code = 1
    try:
        receipt = main()
        code = 0
    except Exception as error:
        receipt = {'status': 'failed', 'exit_code': 1, 'error': repr(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    record = write('PREFLIGHT-RECEIPT.json', receipt)
    (OUT / 'EXIT-CODE.txt').write_text(str(code) + '\n')
    print(json.dumps({'receipt': record, 'status': receipt['status'], 'exit_code': code}), flush=True)
    sys.exit(code)
