"""Verify imported repository files against actual Pod package manifests."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
TREE = HERE / 'source-tree'
BASE = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/technical-prep')
ORIGINAL = BASE / 'resume9010-A-r92-prep-20261028-attempt3'
assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
assert all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

started = time.time()
manifest = json.loads((ORIGINAL / 'PACKAGE-MANIFEST.json').read_text())
checks = []
checks.append({'name': 'package_manifest_exact', 'passed': sha(TREE / 'records/R92/PACKAGE-MANIFEST.json') == sha(ORIGINAL / 'PACKAGE-MANIFEST.json')})
checks.append({'name': 'cpu_receipt_exact', 'passed': sha(TREE / 'records/R92/R92-CPU-RECEIPT.json') == sha(ORIGINAL / 'result/R92-CPU-RECEIPT.json')})
for relative, digest in manifest['files'].items():
    imported = TREE / 'runtime' / relative
    original = ORIGINAL / 'package' / relative
    checks.append({'name': 'runtime/' + relative, 'passed': imported.is_file() and not imported.is_symlink() and sha(imported) == digest == sha(original), 'sha256': digest})
actual_runtime = {str(p.relative_to(TREE / 'runtime')) for p in (TREE / 'runtime').rglob('*') if p.is_file()}
checks.append({'name': 'runtime_file_set_exact', 'passed': actual_runtime == set(manifest['files'])})
checks.append({'name': 'activation_exact', 'passed': sha(TREE / 'tools/activation/activate.py') == manifest['activate_script_sha256'] == sha(ORIGINAL / 'activate_r92.py')})
for relative in ('preflight.py', 'generate_split.py', 'verify_stage1.py'):
    original = BASE / 'resume9010-A-stage1-20261028-attempt1' / relative
    checks.append({'name': 'data_contract/' + relative, 'passed': sha(TREE / 'tools/data-contract' / relative) == sha(original)})
for relative, original in [('PACKAGE-MANIFEST.json', BASE / 'resume9010-A-stage3-20261028-attempt2/PACKAGE-MANIFEST.json'),
                           ('FINAL-RECEIPT.json', BASE / 'resume9010-A-stage4-finalize-20261028-attempt1/result/FINAL-RECEIPT.json')]:
    checks.append({'name': 'R91/' + relative, 'passed': sha(TREE / 'records/R91' / relative) == sha(original)})
forbidden = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.csv', '.tsv', '.jpg', '.jpeg', '.png', '.npy', '.npz', '.zip', '.gz', '.tar'}
files = sorted(p for p in TREE.rglob('*') if p.is_file())
checks.append({'name': 'no_data_models_archives_or_symlinks', 'passed': all(not p.is_symlink() and p.suffix.lower() not in forbidden and '.git' not in p.parts for p in files)})
file_manifest = [{'path': str(p.relative_to(TREE)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
with (HERE / 'REPOSITORY-FILE-MANIFEST.json').open('x') as stream:
    json.dump({'files': file_manifest}, stream, indent=2); stream.write('\n')
receipt = {'status': 'passed' if all(x['passed'] for x in checks) else 'failed',
           'exit_code': 0 if all(x['passed'] for x in checks) else 1, 'started_at': started,
           'finished_at': time.time(), 'checks': checks, 'source_digest': manifest['bundle_source_digest'],
           'model_loaded': False, 'official_forwards': 0, 'locks_acquired': False,
           'repository_files': len(files), 'file_manifest_sha256': sha(HERE / 'REPOSITORY-FILE-MANIFEST.json')}
with (HERE / 'SOURCE-VERIFY-RECEIPT.json').open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
print(json.dumps({'status': receipt['status'], 'exit_code': receipt['exit_code'], 'check_count': len(checks),
                  'failed': [x for x in checks if not x['passed']], 'repository_files': len(files),
                  'receipt_path': str(HERE / 'SOURCE-VERIFY-RECEIPT.json')}), flush=True)
sys.exit(receipt['exit_code'])
