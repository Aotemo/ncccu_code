"""Run selected CPU checks in a verified Pod and keep the actual output and hashes."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pattern', action='append')
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    if 'technical-prep' not in root.parts or not output.is_relative_to(root.parent):
        raise ValueError('CPU checks and their output require an isolated technical-prep directory')
    if socket.gethostname() != os.environ['NCCCU_POD_NAME']:
        raise ValueError('Pod hostname differs from the freshly checked channel')
    if os.path.realpath(sys.executable) != os.path.realpath(os.environ['NCCCU_PYTHON']):
        raise ValueError('interpreter differs from the freshly checked channel')
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('CPU preparation must run at nice 19')
    for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        if os.environ.get(name) != '':
            raise ValueError('CPU preparation must hide all accelerators')
    threads = os.environ.get('NCCCU_CPU_THREADS', '')
    mask = json.loads(os.environ.get('NCCCU_CPU_MASK', 'null'))
    if (not threads.isdecimal() or int(threads) <= 0 or not isinstance(mask, list)
            or len(mask) != int(threads) or sorted(mask) != sorted(os.sched_getaffinity(0))
            or any(os.environ.get(name) != threads for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('CPU preparation requires the explicit allocated mask and thread budget')
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted([*(root / 'runtime').rglob('*.py'), *(root / 'tests').glob('test_*.py'), *(root / 'tools').rglob('*.py')])
    files = {str(path.relative_to(root)): sha(path) for path in paths}
    for path in paths:
        ast.parse(path.read_text(), filename=str(path))
    selected = sorted({path.stem for pattern in (args.pattern or ['test_*.py'])
                       for path in (root / 'tests').glob(pattern) if path.is_file()})
    if not selected:
        raise ValueError('explicit check selection is empty')
    bootstrap = ('import os,sys,torch,unittest;sys.path.insert(0,"tests");torch.set_num_threads(int(os.environ["NCCCU_CPU_THREADS"]));'
                 'torch.set_num_interop_threads(1);assert not torch.cuda.is_initialized();unittest.main(module=None)')
    command = [sys.executable, '-B', '-c', bootstrap, *selected, '-v']
    start = time.time()
    with (output / 'OUTPUT.txt').open('x') as stream:
        result = subprocess.run(command, cwd=root, env=os.environ.copy(), stdout=stream,
                                stderr=subprocess.STDOUT, timeout=300)
    after = {str(path.relative_to(root)): sha(path) for path in paths}
    receipt = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'compute_pod_cpu',
               'pod': socket.gethostname(), 'uid': os.environ['NCCCU_POD_UID'],
               'container': os.environ['NCCCU_CONTAINER'], 'python': sys.executable,
               'command': command, 'cwd': str(root), 'nice': 19, 'threads': int(threads),
               'cpu_mask': mask, 'cpu_allocation': os.environ['NCCCU_CPU_ALLOCATION'],
               'executor_id': os.environ['NCCCU_EXECUTOR_ID'],
               'gpu_visibility': {name: os.environ[name] for name in
                                  ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
               'exit_code': result.returncode, 'source_files': files,
               'syntax_files_checked': len(paths), 'selected_test_modules': selected,
               'source_unchanged_during_checks': files == after,
               'output_sha256': sha(output / 'OUTPUT.txt'), 'wall_seconds': time.time() - start}
    with (output / 'RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print((output / 'OUTPUT.txt').read_text(), end='')
    print(json.dumps({'receipt': str(output / 'RECEIPT.json'),
                      'receipt_sha256': sha(output / 'RECEIPT.json'), 'exit_code': result.returncode,
                      'source_unchanged_during_checks': files == after}, indent=2))
    return result.returncode if files == after else 2


if __name__ == '__main__':
    raise SystemExit(main())
