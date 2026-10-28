"""Read-only storage inventory. Run only on the pinned compute Pod CPU."""
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time

ROOT = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2')
HERE = Path(__file__).resolve().parent
assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12'
assert os.getpriority(os.PRIO_PROCESS, 0) == 19
assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
assert all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))

def write(name, obj):
    with (HERE / name).open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False)
        stream.write('\n')

started = time.time()
large = []
metadata = []
errors = []
skip = {'.git', '__pycache__', 'node_modules', 'site-packages', 'data', 'datasets',
        'images', 'train', 'valid', 'val', 'sealed', 'cpu-env', '.venv', 'venv'}
suffixes = {'.pt', '.pth', '.safetensors', '.ckpt', '.bin', '.tar', '.gz', '.zip', '.npy', '.npz', '.h5'}
metadata_names = {'SINGLE-MODEL-FREEZE.json', 'OFFICIAL-RESULT.json', 'OFFICIAL-ONCE.json',
                  'complete.json', 'done.json', 'best.json', 'latest.json', 'FAILURE.json',
                  'FREEZE.json', 'ASSETS.json', 'TRAIN_CONFIG.json', 'RUNTIME_BINDING.json',
                  'INITIAL-STATE.json', 'REUSED-PASSED-EVIDENCE.json'}
campaigns = [p for p in ROOT.iterdir() if p.is_dir() and (p.name.startswith('auroc') or p.name == 'shared-assets')]
for campaign in sorted(campaigns):
    print(json.dumps({'event': 'scanning', 'path': str(campaign)}), flush=True)
    for base, dirs, files in os.walk(campaign, followlinks=False, onerror=lambda e: errors.append(str(e))):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(('prepared', 'image-cache'))]
        directory = Path(base)
        for name in files:
            path = directory / name
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                if info.st_size >= 8 * 1024**2 and (path.suffix.lower() in suffixes or 'checkpoint' in name.lower()):
                    large.append({'path': str(path), 'bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512,
                                  'inode': info.st_ino, 'device': info.st_dev, 'nlink': info.st_nlink,
                                  'mtime_ns': info.st_mtime_ns})
                if name in metadata_names and info.st_size <= 4 * 1024**2:
                    try:
                        value = json.loads(path.read_text())
                        metadata.append({'path': str(path), 'content': value})
                    except Exception as exc:
                        errors.append(str(path) + ': ' + repr(exc))
            except FileNotFoundError:
                errors.append('disappeared: ' + str(path))

open_files = []
denied = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        for fd in (proc / 'fd').iterdir():
            try:
                target = os.readlink(fd)
                if target.startswith(str(ROOT) + '/'):
                    open_files.append({'pid': int(proc.name), 'fd': fd.name, 'path': target})
            except (FileNotFoundError, ProcessLookupError):
                pass
            except PermissionError:
                denied.append(str(fd))
    except (FileNotFoundError, ProcessLookupError):
        pass
    except PermissionError:
        denied.append(str(proc))

children = []
for child in sorted(ROOT.iterdir()):
    info = child.lstat()
    children.append({'path': str(child), 'is_directory': stat.S_ISDIR(info.st_mode),
                     'is_symlink': stat.S_ISLNK(info.st_mode), 'bytes': info.st_size})
usage = {}
try:
    run = subprocess.run(['du', '-x', '-B1', '--max-depth=1', str(ROOT)],
                         capture_output=True, text=True, timeout=50, check=False)
    usage = {'exit_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr}
except subprocess.TimeoutExpired as exc:
    usage = {'status': 'timeout', 'partial_stdout': (exc.stdout or b'').decode(errors='replace')}

result = {'status': 'read_only_inventory', 'started_at': started, 'finished_at': time.time(),
          'root': str(ROOT), 'hostname': socket.gethostname(), 'python': sys.executable,
          'gpu_hidden': True, 'thread_limit': 2, 'nice': 19, 'large_files': large,
          'metadata': metadata, 'open_files': open_files, 'process_scan_denied': denied,
          'root_children': children, 'disk_usage': usage, 'errors': errors,
          'deleted_files': [], 'training_signals': [], 'locks_acquired': []}
write('STORAGE-INVENTORY.json', result)
summary = {'status': result['status'], 'large_file_count': len(large),
           'large_logical_bytes': sum(f['bytes'] for f in large),
           'metadata_count': len(metadata), 'open_files_count': len(open_files),
           'process_scan_denied_count': len(denied), 'errors': errors,
           'largest_files': sorted(large, key=lambda x: x['bytes'], reverse=True)[:35],
           'root_children': children, 'disk_usage': usage}
write('STORAGE-SUMMARY.json', summary)
print(json.dumps(summary, ensure_ascii=False), flush=True)
