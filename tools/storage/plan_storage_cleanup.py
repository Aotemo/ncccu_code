"""Plan immutable binary deduplication on Pod; preserve every model pathname."""
import collections
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2')
LEGACY = ROOT / 'auroc998-20261025'
assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
assert all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
try:
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
except FileNotFoundError:
    pass

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.resolve() != path:
        raise ValueError('not an exact regular pathname: ' + str(path))
    return {'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size,
            'mtime_ns': info.st_mtime_ns, 'mode': stat.S_IMODE(info.st_mode),
            'uid': info.st_uid, 'gid': info.st_gid, 'nlink': info.st_nlink,
            'allocated_bytes': info.st_blocks * 512}

started = time.time()
inventory = json.loads((HERE / 'STORAGE-INVENTORY.json').read_text())
assert not inventory['errors'] and not inventory['process_scan_denied']
groups = collections.defaultdict(list)
for entry in inventory['large_files']:
    path = Path(entry['path'])
    if path.suffix in ('.pt', '.pth', '.safetensors', '.bin'):
        groups[(path.name, entry['bytes'])].append(path)
groups = [paths for paths in groups.values() if len(paths) > 1 and any(p.is_relative_to(LEGACY) for p in paths)]
plans = []
skipped = []
hashed = {}
for number, paths in enumerate(groups, 1):
    print(json.dumps({'event': 'hashing_duplicate_name_group', 'number': number, 'total': len(groups),
                      'name': paths[0].name, 'copies': len(paths)}), flush=True)
    by_digest = collections.defaultdict(list)
    for path in sorted(paths):
        before = identity(path)
        if before['mtime_ns'] > (started - 600) * 1e9:
            skipped.append({'path': str(path), 'reason': 'recent_file'}); continue
        key = (before['device'], before['inode'])
        if key not in hashed:
            hashed[key] = sha(path)
        after = identity(path)
        assert before == after, str(path)
        by_digest[hashed[key]].append((path, before))
    for digest, copies in by_digest.items():
        if len(copies) < 2:
            continue
        # Prefer an untouched current-campaign or shared canonical original.
        copies.sort(key=lambda x: (x[0].is_relative_to(LEGACY), len(str(x[0])), str(x[0])))
        canonical, source = copies[0]
        for target, original in copies[1:]:
            if not target.is_relative_to(LEGACY):
                continue
            if original['device'] != source['device'] or original['inode'] == source['inode']:
                continue
            if any(original[k] != source[k] for k in ('mode', 'uid', 'gid')):
                skipped.append({'path': str(target), 'reason': 'metadata_differs'}); continue
            if original['nlink'] != 1:
                skipped.append({'path': str(target), 'reason': 'already_shared_inode'}); continue
            if any(word in str(target).lower() for word in ('failure', 'failed', 'rejected')):
                skipped.append({'path': str(target), 'reason': 'failure_original'}); continue
            plans.append({'action': 'replace_duplicate_with_hardlink', 'target': str(target),
                          'canonical': str(canonical), 'sha256': digest, 'target_identity': original,
                          'canonical_identity': source,
                          'reason': 'identical immutable bytes; retain both paths, indexes and selected models'})

usage = []
for path in (LEGACY, ROOT / 'shared-assets'):
    try:
        run = subprocess.run(['du', '-x', '-B1', '--max-depth=2', str(path)], capture_output=True,
                             text=True, timeout=50, check=False)
        usage.append({'root': str(path), 'exit_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr})
    except subprocess.TimeoutExpired:
        usage.append({'root': str(path), 'status': 'timeout'})
plan = {'status': 'read_only_cleanup_plan', 'started_at': started, 'finished_at': time.time(),
        'allowed_target_root': str(LEGACY), 'actions': plans, 'skipped': skipped,
        'candidate_reclaim_allocated_bytes': sum(p['target_identity']['allocated_bytes'] for p in plans),
        'dedup_preserves_all_paths': True, 'active_campaign_targets': [], 'directory_usage': usage,
        'model_loaded': False, 'locks_acquired': False, 'signals_sent': False}
with (HERE / 'STORAGE-CLEANUP-PLAN.json').open('x') as stream:
    json.dump(plan, stream, indent=2); stream.write('\n')
summary = {'status': plan['status'], 'actions': len(plans),
           'candidate_reclaim_allocated_bytes': plan['candidate_reclaim_allocated_bytes'],
           'hashed_unique_inodes': len(hashed), 'skipped': skipped,
           'sample_actions': plans[:5], 'directory_usage': usage}
with (HERE / 'STORAGE-CLEANUP-PLAN-SUMMARY.json').open('x') as stream:
    json.dump(summary, stream, indent=2); stream.write('\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'directory_usage'}), flush=True)
for item in usage:
    print(item.get('stdout', json.dumps(item)), flush=True)
