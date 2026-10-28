"""Read-only explicit asset/storage inventory; all output stays in technical-prep."""
import argparse
import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_asset(spec, stream, seen):
    path = Path(spec['path'])
    if not path.is_absolute():
        raise ValueError('asset path must be explicit and absolute')
    result = {'id': spec['id'], 'path': str(path), 'exists': path.exists(),
              'sha_recomputed': False, 'files': 0, 'directories': 0, 'symlinks': 0,
              'apparent_bytes': 0, 'new_unique_inode_bytes': 0, 'hardlink_entries': 0}
    if not path.exists() and not path.is_symlink():
        result['error'] = 'missing'
        return result

    def observe(item):
        info = item.lstat()
        kind = 'symlink' if stat.S_ISLNK(info.st_mode) else ('directory' if stat.S_ISDIR(info.st_mode) else 'file')
        record = {'asset': spec['id'], 'path': str(item), 'kind': kind, 'bytes': info.st_size,
                  'device': info.st_dev, 'inode': info.st_ino, 'nlink': info.st_nlink,
                  'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}
        if kind == 'symlink':
            result['symlinks'] += 1
            record['target'] = os.readlink(item)
        elif kind == 'directory':
            result['directories'] += 1
        elif stat.S_ISREG(info.st_mode):
            result['files'] += 1
            result['apparent_bytes'] += info.st_size
            result['hardlink_entries'] += int(info.st_nlink > 1)
            key = info.st_dev, info.st_ino
            if key not in seen:
                result['new_unique_inode_bytes'] += info.st_size
                seen.add(key)
        else:
            raise ValueError('non-regular asset object: ' + str(item))
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    observe(path)
    if path.is_dir() and not path.is_symlink():
        def fail(error):
            raise error
        for base, dirs, files in os.walk(path, followlinks=False, onerror=fail):
            for name in sorted(dirs + files):
                observe(Path(base) / name)
    elif path.is_file() and not path.is_symlink() and spec.get('hash') is True:
        result.update(sha256=sha(path), sha_recomputed=True)
        if spec.get('expected_sha256') is not None:
            result['sha_matches'] = result['sha256'] == spec['expected_sha256']
            if not result['sha_matches']:
                result['error'] = 'SHA differs'
    elif spec.get('expected_sha256') is not None:
        result['expected_sha256_unverified'] = spec['expected_sha256']
    if result['symlinks']:
        result['migration_admission'] = 'needs_explicit_symlink_resolution'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.asset_manifest = args.asset_manifest.resolve(strict=True)
    if (socket.gethostname() != os.environ['NCCCU_POD_NAME']
            or os.path.realpath(sys.executable) != os.path.realpath(os.environ['NCCCU_PYTHON'])
            or os.getpriority(os.PRIO_PROCESS, 0) != 19
            or any(os.environ.get(key) != '' for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or not os.environ.get('NCCCU_CPU_THREADS', '').isdecimal()
            or int(os.environ['NCCCU_CPU_THREADS']) <= 0
            or any(os.environ.get(key) != os.environ['NCCCU_CPU_THREADS'] for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
            or sorted(json.loads(os.environ.get('NCCCU_CPU_MASK', '[]'))) != sorted(os.sched_getaffinity(0))):
        raise ValueError('verified Pod CPU channel/nice19/allocated-mask/hidden-GPU contract required')
    output = args.output.resolve()
    if not args.work_root.is_absolute() or not output.is_relative_to(args.work_root.resolve() / 'technical-prep'):
        raise ValueError('inventory output must stay in the explicit work_root/technical-prep')
    specs = json.loads(args.asset_manifest.read_text())['assets']
    if not specs or len({spec['id'] for spec in specs}) != len(specs):
        raise ValueError('nonempty unique asset identities required')
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    results, seen = [], set()
    with (output / 'FILES.jsonl').open('x') as stream:
        for spec in specs:
            try:
                results.append(inventory_asset(spec, stream, seen))
            except (OSError, ValueError) as error:
                results.append({'id': spec['id'], 'path': spec['path'], 'error': repr(error)})
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = 'missing'
    report = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'verified_compute_pod_cpu_inventory', 'pod': socket.gethostname(),
              'namespace': os.environ['NCCCU_POD_NAMESPACE'], 'uid': os.environ['NCCCU_POD_UID'],
              'container': os.environ['NCCCU_CONTAINER'], 'python': sys.executable,
              'nice': 19, 'threads': int(os.environ['NCCCU_CPU_THREADS']), 'dependencies': versions,
              'cpu_allocation': os.environ['NCCCU_CPU_ALLOCATION'], 'executor_id': os.environ['NCCCU_EXECUTOR_ID'],
              'cpu_mask': json.loads(os.environ['NCCCU_CPU_MASK']),
              'gpu_initialized_by_inventory': False, 'model_loaded': False,
              'training_lock_acquired': False, 'signals_sent': False,
              'asset_manifest_sha256': sha(args.asset_manifest), 'script_sha256': sha(__file__),
              'file_inventory_sha256': sha(output / 'FILES.jsonl'), 'assets': results,
              'disk': dict(zip(('total', 'used', 'free'), shutil.disk_usage(args.work_root))),
              'exit_code': int(any('error' in item for item in results))}
    with (output / 'INVENTORY.json').open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False); stream.write('\n')
    print(json.dumps({'receipt': str(output / 'INVENTORY.json'), 'receipt_sha256': sha(output / 'INVENTORY.json'),
                      'exit_code': report['exit_code'], 'assets': results, 'disk': report['disk']}, ensure_ascii=False, indent=2))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
