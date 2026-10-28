"""Manifest-based historical checkpoint and redundant artifact pruning on Pod CPU."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2')
BASE = ROOT / 'auroc999-20261027'
ACTIVE = BASE / 'rounds/R92-TINY1K-9010'
RETAINED = ROOT / 'retained-models'
PLAN = HERE / 'PRUNE3-PLAN.json'
CACHE = ROOT / 'auroc998-20261025/coordination-huazhong8-20261025/r24/prepared768-v2-pillow12.3.0'
assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
assert all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False); stream.write('\n')

def ident(path):
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and path.resolve() == path, str(path)
    return {'device': s.st_dev, 'inode': s.st_ino, 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'allocated_bytes': s.st_blocks * 512, 'nlink': s.st_nlink}

def protected(path):
    if path.is_relative_to(ACTIVE) or path.is_relative_to(HERE) or path.is_relative_to(RETAINED):
        return True
    relative = path.relative_to(ROOT)
    if path.is_relative_to(BASE / 'rounds/R91-R50-IMAGENETV2-9010/control'):
        return True  # Current R92 predecessor metadata, until its final receipt.
    return any(p.startswith(('resume9010-A-r92-', 'resume9010-A-stage1-')) for p in relative.parts)

def cache_identity():
    s = CACHE.lstat()
    assert stat.S_ISDIR(s.st_mode) and CACHE.resolve() == CACHE
    count = total = allocated = 0
    for base, dirs, files in os.walk(CACHE, followlinks=False):
        for name in dirs:
            assert not (Path(base) / name).is_symlink()
        for name in files:
            value = ident(Path(base) / name)
            count += 1; total += value['bytes']
            if value['nlink'] == 1:
                allocated += value['allocated_bytes']
    return {'device': s.st_dev, 'inode': s.st_ino, 'mtime_ns': s.st_mtime_ns,
            'files': count, 'bytes': total, 'single_link_allocated_bytes': allocated}

def scan_open():
    files = set(); active_paths = set(); denied = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] in ('Z', 'X'):
                continue
            for arg in (proc / 'cmdline').read_bytes().split(b'\0'):
                value = arg.decode(errors='replace')
                if value.startswith(str(ROOT) + '/'):
                    active_paths.add(value)
            for fd in (proc / 'fd').iterdir():
                try:
                    value = os.readlink(fd)
                    if value.startswith(str(ROOT) + '/'):
                        files.add(value)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            denied.append(str(proc))
    assert not denied, denied
    return files, active_paths

def make_conclusions():
    if (RETAINED / 'EXPERIMENT-CONCLUSIONS.json').is_file():
        old = read(RETAINED / 'EXPERIMENT-CONCLUSIONS.json')
        assert old['count'] == len(old['experiments']) and old['count'] > 0
        assert (RETAINED / 'EXPERIMENT-CONCLUSIONS.md').is_file()
        return old['count']
    data = read(HERE / 'EXPERIMENT-CONCLUSIONS-v2.json')
    rows = []
    seen = set()
    text = ['# 历史实验结论', '', '历史8:2与当前9:1不同；官方与内部dev分列。缺失或技术失败不视为达标。融合仅保留历史结论，不恢复训练。', '']
    for entry in data['entries']:
        path = Path(entry['path'])
        obj = read(path)
        key = obj.get('round_id') or obj.get('experiment') or path.name
        if not isinstance(key, str):
            key = path.name
        if key in seen:
            continue
        seen.add(key)
        metrics = [m for m in entry['metrics'] if m['metric'].get('count') in (5140, 10277, 5849)
                   and not any(x in m['location'].lower() for x in ('strata', 'reference', 'compar', 'bootstrap', 'delta'))]
        concise = []
        for m in metrics:
            loc = m['location']
            if any(x in loc.lower() for x in ('best', 'dev5k/metrics', 'official', 'valid/metrics', 'valid/independent_metrics')) or loc in ('/metrics', '/independent_metrics', '/valid_metrics'):
                concise.append({'location': loc, 'count': m['metric'].get('count'),
                                'auroc': m['metric'].get('auroc'), 'ap': m['metric'].get('ap')})
        row = {'id': key, 'status': obj.get('status', 'recorded'), 'model': obj.get('model'),
               'successful_updates': obj.get('successful_updates'), 'skipped_updates': obj.get('skipped_updates'),
               'metrics': concise[:12], 'source_result_sha256': sha(path)}
        rows.append(row)
        text += ['## ' + key, '', '状态：' + str(row['status']) + '；模型：' + str(row['model'] or '见指标/原实验定义') + '。', '']
        for m in row['metrics']:
            text.append('- ' + m['location'] + '：n=' + str(m['count']) + '，AUROC=' + str(m['auroc']) + '，AP=' + str(m['ap']) + '。')
        if not row['metrics']:
            text.append('- 没有可归入本表的完整官方/内部选点指标；保留真实状态，不补造数值。')
        md = path.with_suffix('.md')
        if md.is_file() and md.name.startswith('RESULT-'):
            paragraphs = [p.strip() for p in md.read_text().split('\n\n') if p.strip()]
            # Preserve scientific conclusion paragraphs rather than process ledgers.
            chosen = [p for p in paragraphs if not any(w in p for w in ('完整19份', '完整17份', '完整21份', '完整25份', '原始CSV', '原GPU阶段', '原始模型与缓存', '完整CP', '全部17 CSV'))]
            for p in chosen[:5]:
                text += ['', p[:1800]]
        text.append('')
    inventory = read(HERE / 'STORAGE-INVENTORY.json')
    for item in inventory['metadata']:
        p = Path(item['path']); value = item['content']
        if p.name not in ('complete.json', 'FAILURE.json') or protected(p):
            continue
        key = str(p.parent.relative_to(ROOT))
        if key in seen:
            continue
        seen.add(key)
        row = {'id': key, 'status': value.get('status', 'technical_failure' if p.name == 'FAILURE.json' else 'training_completion_record'),
               'successful_updates': value.get('successful_updates'), 'skipped_updates': value.get('skipped_updates'),
               'conclusion': 'Recorded training/technical terminal state; no new official metric inferred.',
               'error': str(value.get('error', value.get('exception', value.get('reason', ''))))[:1200]}
        rows.append(row)
        text += ['## ' + key, '', '状态：' + str(row['status']) + '；成功更新：' + str(row['successful_updates'])
                 + '；skip：' + str(row['skipped_updates']) + '。官方指标未在该终态中确认，不补造数值。', '']
        if row['error']:
            text += [row['error'], '']
    write(RETAINED / 'EXPERIMENT-CONCLUSIONS.json', {'experiments': rows, 'count': len(rows)})
    with (RETAINED / 'EXPERIMENT-CONCLUSIONS.md').open('x') as stream:
        stream.write('\n'.join(text) + '\n')
    return len(rows)

def plan():
    keep = read(HERE / 'KEEP-MODELS.json')
    model_paths = {Path(m['path']) for m in keep['models']}
    for model in keep['models']:
        p = Path(model['path']); assert ident(p)['bytes'] > 0 and sha(p) == model['sha256'], model['id']
    count = make_conclusions()
    prior_keep = RETAINED / 'KEEP-MODELS.json'
    if prior_keep.exists():
        prior_keep.unlink()
    write(prior_keep, keep)
    if (RETAINED / 'R91').exists():
        shutil.rmtree(RETAINED / 'R91')  # Superseded teaching exception; small metadata only.
    # Keep a minimal, self-contained identity/configuration note for each exact blob.
    for model in keep['models']:
        p = Path(model['path'])
        nearby = []
        for parent in (p.parent.parent, p.parent.parent.parent):
            for name in ('contract.json', 'identity.json', 'source-files.json', 'config.json', 'job.json'):
                source = parent / name
                if source.is_file() and source.stat().st_size < 2 * 1024**2:
                    dest = RETAINED / model['id'] / (parent.name + '-' + name)
                    dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
                    nearby.append(str(dest))
        model_note = RETAINED / model['id'] / 'MODEL.json'
        if model_note.exists():
            model_note.unlink()
        write(model_note, {**model, 'minimal_records': nearby,
              'loading': 'Use the exact checkpoint file with explicit variant=ema; payload retains job, age, normalization and identity. Do not resume an obsolete checkpoint index.'})
    open_files, active_paths = scan_open()
    actions = []
    artifact_parts = {'outputs', 'audits', 'audit', 'evidence', 'predictions', 'evaluation', 'analysis', 'control'}
    skip = {'.git', '__pycache__', 'site-packages', 'data', 'datasets', 'images', 'train', 'valid', 'sealed', 'node_modules', '.venv', 'venv'}
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        directory = Path(base)
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('prepared')
                   and not protected(directory / d)]
        for name in files:
            p = directory / name
            if protected(p) or p.is_symlink():
                continue
            if p in model_paths or any(p.parent == k.parent and p.name.startswith(k.stem) for k in model_paths):
                continue
            parts = p.relative_to(ROOT).parts
            is_checkpoint = (bool(re.fullmatch(r'checkpoint-[0-9a-f]+\.(pt|pth|ckpt)', name))
                             or p.suffix in ('.pt', '.pth', '.ckpt', '.safetensors')
                             and (any(s in ('checkpoints', 'initial-checkpoint') or s.startswith('CPU-FIXTURE') for s in parts)
                                  or name in ('best.pt', 'last.pt', 'model.pt', 'model.pth')))
            is_document = p.suffix.lower() == '.md' and name not in ('AGENTS.md', 'SSH-START.md')
            in_artifacts = any(s in artifact_parts for s in parts) and any(s.startswith('auroc') for s in parts)
            artifact = in_artifacts and (p.suffix.lower() in ('.csv', '.tsv', '.log', '.jsonl', '.png', '.jpg', '.jpeg', '.npz', '.npy', '.zip', '.gz', '.tar')
                                         or p.suffix == '.json' and not any(s in ('assets', 'weights', 'public-weights', 'source', 'converted') for s in parts))
            # Raw package/config JSON is source, not a per-run artifact.
            if not (is_checkpoint or is_document or artifact):
                continue
            if str(p) in open_files or str(p) in active_paths:
                raise ValueError('planned target is in use: ' + str(p))
            actions.append({'path': str(p), 'identity': ident(p),
                            'kind': 'checkpoint' if is_checkpoint else 'document' if is_document else 'discarded_process_artifact'})
    if CACHE.is_dir():
        assert not any(str(p).startswith(str(CACHE) + '/') for p in open_files | active_paths)
        actions.append({'path': str(CACHE), 'identity': cache_identity(), 'kind': 'unused_prepared768_cache'})
    value = {'status': 'ready_for_explicit_apply', 'created_at': time.time(), 'policy': keep['policy'],
             'retained_models_verified': keep['models'], 'pending_teaching': keep['pending_teaching'],
             'conclusion_count': count, 'actions': actions, 'protected_active_root': str(ACTIVE),
             'delete_logical_bytes': sum(a['identity']['bytes'] for a in actions),
             'kind_counts': {k: sum(a['kind'] == k for a in actions) for k in ('checkpoint', 'document', 'discarded_process_artifact', 'unused_prepared768_cache')}}
    write(PLAN, value)
    summary = {k: v for k, v in value.items() if k != 'actions'}
    summary['plan_sha256'] = sha(PLAN)
    write(HERE / 'PRUNE3-PLAN-SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)

def verify_remaining_identity(path, original, unlinked):
    actual = ident(path)
    inode = (original['device'], original['inode'])
    expected = {**original, 'nlink': original['nlink'] - unlinked[inode]}
    assert expected['nlink'] >= 1 and actual == expected, str(path)
    return actual

def check_hardlink_guard():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='link-guard-', dir=HERE) as folder:
        a = Path(folder) / 'a'; b = Path(folder) / 'b'
        a.write_bytes(b'identity-check'); os.link(a, b)
        original = ident(b); a.unlink()
        removed = collections.Counter({(original['device'], original['inode']): 1})
        verify_remaining_identity(b, original, removed)
        b.write_bytes(b'unexpected-change')
        rejected = False
        try:
            verify_remaining_identity(b, original, removed)
        except AssertionError:
            rejected = True
        assert rejected
    result = {'status': 'passed', 'expected_link_count_decrement_accepted': True,
              'unexpected_content_metadata_change_rejected': True, 'model_loaded': False}
    write(HERE / 'PRUNE3-LINK-GUARD-CHECK.json', result)
    return result

def apply(expected):
    assert sha(PLAN) == expected
    value = read(PLAN)
    keep = {Path(m['path']) for m in value['retained_models_verified']}
    assert (RETAINED / 'EXPERIMENT-CONCLUSIONS.json').is_file()
    open_files, active_paths = scan_open()
    journal = HERE / 'PRUNE3-JOURNAL.jsonl'
    deleted = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    assert len(deleted) <= len(value['actions'])
    unlinked = collections.Counter()
    for action, record in zip(value['actions'], deleted):
        assert all(action[k] == record[k] for k in ('path', 'identity', 'kind'))
        assert not Path(action['path']).exists(), action['path']
        if action['kind'] != 'unused_prepared768_cache':
            unlinked[(action['identity']['device'], action['identity']['inode'])] += 1
    resumed_count = len(deleted)
    # The earlier complete precheck is proven by its nonempty flushed journal.
    # Only link counts changed through this exact recorded prefix may differ.
    for action in value['actions'][resumed_count:]:
        p = Path(action['path'])
        assert p.is_relative_to(ROOT) and p not in keep and not protected(p), str(p)
        assert str(p) not in open_files and str(p) not in active_paths, str(p)
        if action['kind'] == 'unused_prepared768_cache':
            assert p == CACHE and not any(x.startswith(str(CACHE) + '/') for x in open_files | active_paths)
            if not resumed_count:
                assert cache_identity() == action['identity']
            else:
                s = CACHE.lstat()
                assert CACHE.resolve() == CACHE and stat.S_ISDIR(s.st_mode)
                assert all(getattr(s, field) == action['identity'][key] for field, key in
                           (('st_dev', 'device'), ('st_ino', 'inode'), ('st_mtime_ns', 'mtime_ns')))
        else:
            verify_remaining_identity(p, action['identity'], unlinked)
    for model in value['retained_models_verified']:
        assert sha(Path(model['path'])) == model['sha256'], model['id']
    before = shutil.disk_usage(ROOT).free
    guard = check_hardlink_guard()
    print(json.dumps({'status': 'verified_same_plan_resume', 'already_deleted': resumed_count,
                      'remaining_actions': len(value['actions']) - resumed_count, 'link_guard': guard}), flush=True)
    with journal.open('a' if resumed_count else 'x') as stream:
        for action in value['actions'][resumed_count:]:
            p = Path(action['path'])
            if action['kind'] == 'unused_prepared768_cache':
                assert p == CACHE and p.resolve() == p
                s = p.lstat()
                assert stat.S_ISDIR(s.st_mode) and s.st_ino == action['identity']['inode']
                shutil.rmtree(p)
                record = action
            else:
                actual = verify_remaining_identity(p, action['identity'], unlinked)
                p.unlink()
                record = {**action, 'actual_nlink_before_delete': actual['nlink']}
                unlinked[(actual['device'], actual['inode'])] += 1
            stream.write(json.dumps(record) + '\n'); stream.flush()
            deleted.append(record)
    remaining = [a['path'] for a in value['actions'] if Path(a['path']).exists()]
    assert not remaining
    for model in value['retained_models_verified']:
        assert sha(Path(model['path'])) == model['sha256'], model['id']
    receipt = {'status': 'completed', 'exit_code': 0, 'plan_sha256': expected, 'deleted_files': len(deleted),
               'deleted_logical_bytes': sum(a['identity']['bytes'] for a in deleted),
               'freed_allocated_bytes_from_single_link_files': sum(a['identity']['allocated_bytes'] for a in deleted if a.get('actual_nlink_before_delete', a['identity'].get('nlink')) == 1)
                    + sum(a['identity']['single_link_allocated_bytes'] for a in deleted if a['kind'] == 'unused_prepared768_cache'),
               'free_bytes_before_this_resume': before, 'free_bytes_after': shutil.disk_usage(ROOT).free,
               'resumed_journal_prefix_count': resumed_count, 'link_guard_check': guard,
               'cleanup_script_sha256': sha(Path(__file__)),
               'kind_counts': value['kind_counts'], 'retained_models_verified': value['retained_models_verified'],
               'pending_teaching': value['pending_teaching'], 'conclusion_count': value['conclusion_count'],
               'active_root_untouched': str(ACTIVE), 'locks_acquired': False, 'signals_sent': False, 'model_loaded': False}
    write(HERE / 'PRUNE3-RECEIPT.json', receipt)
    write(RETAINED / 'PRUNE-RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('plan', 'apply'))
parser.add_argument('--plan-sha256')
args = parser.parse_args()
if args.phase == 'plan':
    plan()
else:
    assert args.plan_sha256 and re.fullmatch('[0-9a-f]{64}', args.plan_sha256)
    apply(args.plan_sha256)
