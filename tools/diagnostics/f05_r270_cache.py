"""Read-only revalidation of the registered R270 fit/dev 768 cache."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time

HISTORICAL_SHA256 = '605569da9823b2eb445f924a82ee4615f31ed2785d695ea6f866a78a27c2e251'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_index(manifest, index, expected_count):
    """Bind every fit/dev row to exactly one historical cache index entry."""
    if (type(expected_count) is not int or expected_count < 1 or not isinstance(manifest, dict)
            or not isinstance(index, list) or len(index) != expected_count):
        raise ValueError('explicit complete cache index required')
    rows = manifest.get('rows'); fit = manifest.get('fit_indices'); dev = manifest.get('dev_indices')
    if (not isinstance(rows, list) or len(rows) != expected_count
            or not isinstance(fit, list) or not isinstance(dev, list) or not fit or not dev
            or any(type(i) is not int for i in fit + dev)
            or len(set(fit)) != len(fit) or len(set(dev)) != len(dev)
            or set(fit) & set(dev) or sorted(fit + dev) != list(range(expected_count))):
        raise ValueError('complete disjoint fit/dev roles required')
    for number, (row, entry) in enumerate(zip(rows, index)):
        if not isinstance(row, dict) or not isinstance(entry, dict):
            raise ValueError('invalid manifest or cache index row')
        relative, source_sha = row.get('relative_path'), row.get('sha256')
        if not isinstance(relative, str) or not _digest(source_sha):
            raise ValueError('sealed fit/dev image identity required')
        path = PurePosixPath(relative)
        if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] != 'train'
                or path.as_posix() != relative or any(part in ('.', '..') for part in path.parts)):
            raise ValueError('only sealed training images are allowed')
        key = _sha((relative + '\0' + source_sha).encode())
        if (entry.get('manifest_index') != number or entry.get('key') != key
                or not _digest(entry.get('receipt_sha256')) or not _digest(entry.get('array_sha256'))):
            raise ValueError('historical cache row identity differs')
    return index


def verify_rows(manifest, entries, cache, workers=1):
    """Read and hash historical sidecars and arrays through the current cache path."""
    if cache.readonly is not True or type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('read-only cache and bounded workers required')

    def verify(entry):
        row = manifest['rows'][entry['manifest_index']]
        sidecar = Path(cache.directory) / entry['key'][:2] / (entry['key'] + '.json')
        if sidecar.is_symlink():
            raise ValueError('cache sidecar cannot be a symlink')
        raw = sidecar.read_bytes()
        if (_sha(raw) != entry['receipt_sha256']
                or json.loads(raw).get('array_sha256') != entry['array_sha256']):
            raise ValueError('historical cache sidecar differs')
        image = cache.read(row['relative_path'], row['sha256'])
        image.close()
        if _sha(sidecar.read_bytes()) != entry['receipt_sha256']:
            raise ValueError('cache sidecar changed during read')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(verify, entries):
            pass
    return len(entries)


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value, replace=False):
    path = Path(path)
    target = path.with_name('.' + path.name + '.tmp') if replace else path
    with target.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


def _core_matches(source, expected):
    trees = {}
    if not isinstance(expected, dict) or not expected:
        raise ValueError('historical cache core identity required')
    for key, digest in expected.items():
        name, node_name = key.split(':')
        if name not in ('runtime/bundle/retrain/data.py', 'runtime/bundle/retrain/preprocessing.py'):
            raise ValueError('unexpected historical cache source')
        if name not in trees:
            trees[name] = ast.parse((source / name).read_text())
        nodes = [node for node in trees[name].body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and node.name == node_name]
        if (len(nodes) != 1 or _sha(ast.dump(nodes[0], include_attributes=False).encode()) != digest):
            raise ValueError('current cache core differs: ' + key)
    return len(expected)


def _blocked_fraction():
    states = []
    for path in Path('/proc/self/task').glob('*/status'):
        try:
            line = next(item for item in path.read_text().splitlines() if item.startswith('State:'))
            states.append(line.split()[1])
        except (OSError, StopIteration, IndexError):
            continue
    return states.count('D') / len(states) if states else 0.0


def _run_batch(manifest, entries, cache, workers):
    stop = threading.Event(); samples = []

    def monitor():
        while not stop.wait(0.5):
            samples.append(_blocked_fraction())

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start(); started = time.monotonic()
    try:
        count = verify_rows(manifest, entries, cache, workers)
    finally:
        stop.set(); thread.join()
    return count, time.monotonic() - started, max(samples, default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--site-sha256', required=True)
    parser.add_argument('--historical-receipt', type=Path, required=True)
    parser.add_argument('--historical-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(k) != '' for k in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or not 1 <= args.max_workers <= min(8, int(os.environ['NCCCU_CPU_THREADS']))
            or not 128 <= args.batch_size <= 2048):
        raise ValueError('guarded hidden-device CPU pool and bounded scan required')
    if (args.site.is_symlink() or args.historical_receipt.is_symlink()
            or args.historical_sha256 != HISTORICAL_SHA256
            or _file_sha(args.site) != args.site_sha256
            or _file_sha(args.historical_receipt) != args.historical_sha256):
        raise ValueError('site or historical receipt identity differs')
    site, historical = (json.loads(path.read_text()) for path in (args.site, args.historical_receipt))
    work = Path(site['work_root']).resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    output = args.output_dir
    if (site['pod']['uid'] != os.environ['NCCCU_POD_UID'] or not output.is_absolute()
            or not output.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')
            or output.exists()):
        raise ValueError('live Pod or fresh technical-prep output differs')
    manifest_path = Path(site['metadata_root']) / 'manifest.json'
    manifest_sha = _file_sha(manifest_path)
    cache_root = work / 'shared-assets/prepared768-global-v1-pillow12.3.0'
    if cache_root.is_symlink() or cache_root.resolve(strict=True) != cache_root:
        raise ValueError('registered cache path differs')
    expected = {'format':'global-prepared-cache-v1', 'passed':True,
                'data_root':site['data_root'], 'metadata_root':site['metadata_root'],
                'cache_root':str(cache_root), 'manifest_sha256':manifest_sha,
                'fit_count':46256, 'dev_count':5140, 'row_count':51396,
                'verified_readonly_count':51396, 'input_preprocessing':None,
                'prepared_size':768, 'pillow':'12.3.0', 'official_read':False}
    if any(historical.get(key) != value for key, value in expected.items()):
        raise ValueError('historical cache manifest, source or policy differs')
    core_count = _core_matches(source, historical.get('cache_core_AST_sha256'))
    reference = historical.get('index', {})
    index_path = Path(reference.get('path', ''))
    if (not index_path.is_absolute() or index_path.is_symlink()
            or not index_path.resolve(strict=True).is_relative_to(work)
            or _file_sha(index_path) != reference.get('sha256')):
        raise ValueError('historical cache index identity differs')
    manifest = json.loads(manifest_path.read_text())
    with index_path.open('r') as stream:
        index = [json.loads(line) for line in stream]
    entries = validate_index(manifest, index, expected_count=51396)
    if len(manifest['fit_indices']) != 46256 or len(manifest['dev_indices']) != 5140:
        raise ValueError('fit/dev counts differ')
    sys.path.insert(0, str(source / 'runtime/bundle'))
    os.environ['TRAIN_IMAGE_CACHE_READONLY'] = '1'
    from PIL import __version__ as pillow_version
    from retrain.preprocessing import PreparedImageCache
    if pillow_version != '12.3.0':
        raise ValueError('Pillow implementation differs')
    cache = PreparedImageCache(site['data_root'], cache_root / 'train/sealed', 768)
    if not cache.readonly:
        raise ValueError('cache cannot be opened writable')
    output.mkdir(exist_ok=False)
    verified = 0; workers = args.max_workers; started = time.time(); batches = []
    try:
        for start in range(0, len(entries), args.batch_size):
            batch = entries[start:start + args.batch_size]
            count, elapsed, blocked = _run_batch(manifest, batch, cache, workers)
            verified += count
            batches.append({'count':count, 'seconds':elapsed, 'workers':workers, 'max_D_fraction':blocked})
            if blocked > 0.25 or elapsed * workers / count > 1.0:
                workers = max(1, workers // 2)
            elif blocked < 0.05 and elapsed * workers / count < 0.1:
                workers = min(args.max_workers, workers + 1)
            _write(output / 'PROGRESS.json', {'status':'running', 'at':time.time(),
                   'verified_readonly_count':verified, 'total':len(entries), 'next_workers':workers,
                   'latest_batch_seconds':elapsed, 'latest_batch_max_D_fraction':blocked}, replace=True)
            if len(batches) % 10 == 0 or verified == len(entries):
                print(json.dumps({'verified':verified, 'total':len(entries), 'workers':workers,
                                  'latest_batch_seconds':elapsed, 'max_D_fraction':blocked}), flush=True)
        if verified != 51396:
            raise ValueError('complete fit/dev cache coverage required')
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            raise ValueError('GPU unexpectedly initialized')
        value = {**expected, 'pod_uid':site['pod']['uid'],
                 'source_files':{name:_file_sha(source / name) for name in
                     ('runtime/bundle/retrain/data.py', 'runtime/bundle/retrain/preprocessing.py')},
                 'index':reference, 'historical_receipt':{'path':str(args.historical_receipt),
                     'sha256':args.historical_sha256}, 'historical_core_nodes_matched':core_count,
                 'site_sha256':args.site_sha256, 'checker_sha256':_file_sha(__file__),
                 'started_at':started, 'at':time.time(), 'batches':batches,
                 'cache_mutated':False, 'GPU_initialized':False}
        receipt = output / 'GLOBAL768-CACHE-HANDOFF.json'
        _write(receipt, value)
        _write(output / 'PROGRESS.json', {'status':'completed', 'at':time.time(),
               'verified_readonly_count':verified, 'total':len(entries),
               'receipt_sha256':_file_sha(receipt)}, replace=True)
        print(json.dumps({'status':'complete_current_pod_full_readonly_768_cache',
                          'verified':verified, 'receipt_sha256':_file_sha(receipt)}), flush=True)
    except BaseException as error:
        _write(output / 'FAILED.json', {'status':'failed_or_interrupted', 'at':time.time(),
               'verified_readonly_count':verified, 'error_type':type(error).__name__,
               'GPU_initialized':False, 'official_read':False})
        raise


if __name__ == '__main__':
    main()
