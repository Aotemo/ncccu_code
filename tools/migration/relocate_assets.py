"""Content-verified relocation of the current inventoried training dependencies.

Old objects and frozen sidecars remain untouched. Bulk copies raw bytes and only
needed prepared arrays; a separate tool writes receipts for the new namespace.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretraining'))
from source_chain import cpu_context, file_identity, read_pinned_receipt, require, sha256, write_new_json


def safe_path(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts, 'explicit safe absolute path required')
    for parent in reversed((path, *path.parents)):
        if parent.exists() or parent.is_symlink():
            require(not parent.is_symlink(), 'symlink route is forbidden: ' + str(parent))
    return path


def hash_current(path):
    path = safe_path(path)
    before = file_identity(path)
    digest, count = hashlib.sha256(), 0
    fields = {'bytes': 'st_size', 'device': 'st_dev', 'inode': 'st_ino',
              'mtime_ns': 'st_mtime_ns', 'ctime_ns': 'st_ctime_ns'}
    with path.open('rb') as stream:
        for observed in (os.fstat(stream.fileno()),):
            require(all(before[k] == getattr(observed, v) for k, v in fields.items()), 'opened object differs')
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block); count += len(block)
        observed = os.fstat(stream.fileno())
        require(all(before[k] == getattr(observed, v) for k, v in fields.items()), 'object changed while reading')
    require(file_identity(path) == before and count == before['bytes'], 'object identity/length changed')
    return digest.hexdigest(), before


def copy_verified_file(source, target, expected_sha=None):
    source, target = safe_path(source), safe_path(target)
    require(source != target and source.resolve() != target.resolve(), 'copy cannot modify source')
    before = file_identity(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_path(target)
    temporary = None
    digest, count = hashlib.sha256(), 0
    try:
        descriptor, temporary = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.copy', dir=target.parent)
        with os.fdopen(descriptor, 'wb') as output, source.open('rb') as incoming:
            opened = os.fstat(incoming.fileno())
            require((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) ==
                    (before['device'], before['inode'], before['bytes'], before['mtime_ns'], before['ctime_ns']),
                    'opened source differs')
            for block in iter(lambda: incoming.read(1048576), b''):
                digest.update(block); output.write(block); count += len(block)
            output.flush(); os.fsync(output.fileno())
            finished = os.fstat(incoming.fileno())
            require((finished.st_dev, finished.st_ino, finished.st_size, finished.st_mtime_ns, finished.st_ctime_ns) ==
                    (before['device'], before['inode'], before['bytes'], before['mtime_ns'], before['ctime_ns']),
                    'source changed while copying')
        actual = digest.hexdigest()
        require(count == before['bytes'] and file_identity(source) == before, 'source identity/length changed')
        require(expected_sha is None or actual == expected_sha, 'source content SHA differs from pinned dependency')
        # No overwrite: identical earlier target is accepted only after content proof.
        reused = target.exists()
        if not reused:
            try:
                os.link(temporary, target)
            except FileExistsError:
                reused = True
        target_sha, target_identity = hash_current(target)
        require(target_sha == actual, 'existing/copied target content SHA differs')
        return {'source_path': str(source), 'target_path': str(target), 'sha256': actual,
                'target_sha256': target_sha, 'size': count, 'source_identity': before,
                'target_identity': target_identity, 'evidence': 'fresh_pod_sha256',
                'source_modified_by_tool': False, 'verified_existing_target': reused}
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def relative_inside(path, root):
    relative = Path(path).relative_to(root)
    require('..' not in relative.parts and not relative.is_absolute(), 'inventory path escapes root')
    return relative


def plan(args, context):
    inventory = read_pinned_receipt(args.inventory, args.inventory_sha256)
    dependencies = read_pinned_receipt(args.dependencies, args.dependencies_sha256)
    public = read_pinned_receipt(args.public_receipt, args.public_receipt_sha256)
    require(inventory['uid'] == dependencies['uid'] == public['execution']['uid'] == context['uid']
            and inventory['exit_code'] == 0 and public['passed'] is True, 'current successful source evidence required')
    require(dependencies['inventory_sha256'] == args.inventory_sha256, 'dependency inventory pin differs')
    files = Path(args.inventory).parent / 'FILES.jsonl'
    require(sha256(files) == inventory['file_inventory_sha256'], 'file inventory pin differs')
    assets = {row['id']: row for row in inventory['assets']}
    root = safe_path(args.work_root)
    require(root.is_dir(), 'existing explicit exp-v3 work root required')
    small_routes = {'fixed_9010_stage1_18_objects': root / 'assets/fixed-9010-stage1',
                    'public_tiny22k_original': root / 'assets/public/tiny22k-bc48a87f/original',
                    'public_tiny22k_converted': root / 'assets/public/tiny22k-bc48a87f/converted'}
    paths = {'work_root': str(root), 'dataset_root': str(root / 'data/glaucoma_first'),
             'train_identity': str(root / 'assets/train_identity.json'),
             'stage1_root': str(small_routes['fixed_9010_stage1_18_objects']),
             'public_original_root': str(small_routes['public_tiny22k_original']),
             'public_converted_root': str(small_routes['public_tiny22k_converted']),
             'public_pretrained': str(small_routes['public_tiny22k_converted'] / 'convnext-tiny-tv-bc48a87f.pt'),
             'public_metadata': str(small_routes['public_tiny22k_converted'] / 'provenance.json'),
             'official_manifest': str(root / 'assets/official-valid-sealed.json'),
             'prepared_cache_root': str(root / 'shared-assets/prepared512-v2-pillow12.3.0')}
    raw_root, cache_root = Path(assets['raw_glaucoma_dataset']['path']), Path(assets['prepared512_pillow123']['path'])
    old_policy = {'version': 'rgb-conservative-border-prepared-v2', 'prepared_size': 512,
                  'pillow': '12.3.0', 'dataset_root': str(raw_root)}
    old_namespace = hashlib.sha256(json.dumps(old_policy, sort_keys=True).encode()).hexdigest()
    new_policy = dict(old_policy, dataset_root=paths['dataset_root'])
    new_namespace = hashlib.sha256(json.dumps(new_policy, sort_keys=True).encode()).hexdigest()
    expected_counts = {}
    for split in ('train', 'valid'):
        group = split + '/sealed/' + old_namespace
        observed = dependencies['cache_groups'][group]
        require(observed['sample']['policy'] == old_policy and observed['files'] % 2 == 0, 'cache policy/count differs')
        expected_counts[split] = observed['files'] // 2
    public_sha = {row['path']: row['sha256'] for row in public['assets'].values()}
    frozen = dependencies['stage1']['frozen_files']
    selected, cache_counts = [], {'train': 0, 'valid': 0}
    with files.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != 'file':
                continue
            asset, source = row['asset'], Path(row['path'])
            target, expected = None, public_sha.get(str(source))
            if args.phase == 'small' and asset in small_routes:
                relative = relative_inside(source, Path(assets[asset]['path']))
                target = small_routes[asset] / relative
                if asset == 'fixed_9010_stage1_18_objects':
                    expected = ('90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6'
                                if str(relative) == 'FREEZE.json' else frozen[str(relative)]['sha256'])
            elif args.phase == 'small' and asset == 'recovered_original_readme':
                target = small_routes['public_tiny22k_original'] / 'README.md'
            elif args.phase == 'small' and asset == 'sealed_official_manifest':
                target = Path(paths['official_manifest'])
            elif args.phase == 'bulk' and asset == 'raw_glaucoma_dataset':
                target = Path(paths['dataset_root']) / relative_inside(source, raw_root)
            elif args.phase == 'bulk' and asset == 'prepared512_pillow123':
                parts = relative_inside(source, cache_root).parts
                if (len(parts) == 5 and parts[0] in cache_counts and parts[1:3] == ('sealed', old_namespace)
                        and source.suffix == '.npy'):
                    target = Path(paths['prepared_cache_root']) / parts[0] / parts[1] / new_namespace / parts[3] / parts[4]
                    cache_counts[parts[0]] += 1
            if target is not None:
                selected.append({'asset': asset, 'source': source, 'target': target, 'expected': expected,
                                 'inventory_identity': row})
    if args.phase == 'small':
        identity = dependencies['train_identity']
        selected.append({'asset': 'train_identity', 'source': Path(identity['path']),
                         'target': Path(paths['train_identity']), 'expected': identity['sha256'],
                         'inventory_identity': {'bytes': identity['bytes']}})
        require(sum(row['asset'] == 'fixed_9010_stage1_18_objects' for row in selected) == 19,
                'exactly 18 frozen files plus original FREEZE required')
    else:
        require(cache_counts == expected_counts, 'inventoried prepared array counts differ')
        require(sum(row['asset'] == 'raw_glaucoma_dataset' for row in selected) == assets['raw_glaucoma_dataset']['files'],
                'raw file inventory count differs')
    require(len({str(row['target']) for row in selected}) == len(selected) and selected, 'duplicate/empty copy plan')
    for row in selected:
        require(row['target'].is_relative_to(root) and not row['source'].is_relative_to(row['target']), 'unsafe target layout')
    return selected, paths, {'old_policy': old_policy, 'new_policy': new_policy,
                             'old_namespace': old_namespace, 'new_namespace': new_namespace,
                             'expected_entries': expected_counts}, inventory['file_inventory_sha256']


def run(args):
    context = cpu_context()
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    selected, paths, cache, inventory_files_sha = plan(args, context)
    predicted = sum(row['inventory_identity']['bytes'] for row in selected)
    free = shutil.disk_usage(args.work_root).free
    require(free >= predicted + 1024 ** 3, 'insufficient target capacity plus 1GiB reserve')
    output = safe_path(args.output)
    require(output.is_relative_to(Path(args.work_root) / 'technical-prep') and not output.exists(), 'new independent receipt required')
    manifest = output.with_suffix('.files.jsonl')
    require(not manifest.exists(), 'new independent file manifest required')
    workers = max(1, context['cpu_allocation']['threads'] - 1)
    completed, copied_bytes, counts = 0, 0, {}
    copy_started = datetime.now(timezone.utc).isoformat()
    write_new_json(output.with_suffix('.started.json'), {'passed': False,
        'scope': 'copy_start_only_not_acceptance', 'phase': args.phase, 'copy_started_at': copy_started,
        'execution': context, 'planned_files': len(selected), 'predicted_bytes': predicted,
        'free_bytes': free, 'workers': workers, 'source_modified_by_tool': False})
    print(json.dumps({'event': 'copy_started', 'at': datetime.now(timezone.utc).isoformat(),
        'phase': args.phase, 'files': len(selected), 'predicted_bytes': predicted, 'free_bytes': free,
        'workers': workers, 'output': str(output)}), flush=True)
    def copy(row):
        actual = file_identity(row['source'])
        require(all(actual[key] == value for key, value in row['inventory_identity'].items()
                    if key in actual), 'inventoried source metadata changed')
        proof = copy_verified_file(row['source'], row['target'], row['expected'])
        return dict(proof, asset=row['asset'])
    try:
        with manifest.open('x') as stream, ThreadPoolExecutor(max_workers=workers) as pool:
            # map submits the fixed, pinned inventory; result order is deterministic.
            for proof in pool.map(copy, selected):
                stream.write(json.dumps(proof, sort_keys=True) + '\n')
                completed += 1; copied_bytes += proof['size']
                group = counts.setdefault(proof['asset'], {'files': 0, 'bytes': 0})
                group['files'] += 1; group['bytes'] += proof['size']
                if completed % 512 == 0 or completed == len(selected):
                    stream.flush()
                    progress = {'event': 'copy_progress', 'phase': args.phase, 'files': completed,
                        'bytes': copied_bytes, 'wall_seconds': time.monotonic() - clock,
                        'copy_started_at': copy_started, 'at': datetime.now(timezone.utc).isoformat()}
                    progress_path = output.with_suffix('.progress.json')
                    fd, temporary_progress = tempfile.mkstemp(prefix='.progress.', dir=output.parent)
                    try:
                        with os.fdopen(fd, 'w') as progress_stream:
                            json.dump(progress, progress_stream)
                        os.replace(temporary_progress, progress_path)
                    finally:
                        Path(temporary_progress).unlink(missing_ok=True)
                    print(json.dumps(progress), flush=True)
            stream.flush(); os.fsync(stream.fileno())
        value = {'schema_version': 1, 'passed': True, 'phase': args.phase, 'execution': context,
            'started_at': started, 'completed_at': datetime.now(timezone.utc).isoformat(),
            'copy_started_at': copy_started,
            'wall_seconds': time.monotonic() - clock, 'files': completed, 'bytes': copied_bytes,
            'groups': counts, 'paths': paths, 'cache': cache, 'workers': workers,
            'manifest': {'path': str(manifest), 'sha256': sha256(manifest)},
            'inputs': {'inventory': {'path': args.inventory, 'sha256': args.inventory_sha256},
                       'file_inventory_sha256': inventory_files_sha,
                       'dependencies': {'path': args.dependencies, 'sha256': args.dependencies_sha256},
                       'public_receipt': {'path': args.public_receipt, 'sha256': args.public_receipt_sha256}},
            'free_bytes_before': free, 'free_bytes_after': shutil.disk_usage(args.work_root).free,
            'source_modified_by_tool': False, 'historical_FREEZE_bytes_preserved': True,
            'official_manifest_semantics_read': False, 'GPU_work': False,
            'implementation_sha256': sha256(__file__)}
        write_new_json(output, value)
    except Exception as error:
        write_new_json(output, {'schema_version': 1, 'passed': False, 'phase': args.phase, 'execution': context,
            'files': completed, 'bytes': copied_bytes, 'started_at': started,
            'error_type': type(error).__name__, 'error': str(error), 'partial_manifest': str(manifest),
            'source_modified_by_tool': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'inventory-sha256', 'dependencies', 'dependencies-sha256',
                 'public-receipt', 'public-receipt-sha256', 'work-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--phase', choices=('small', 'bulk'), required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
