"""Rebind copied prepared512 bytes to a new data-root namespace on the Pod.

The pinned copy proof supplies historical source-byte evidence. This tool reads
only old JSON receipts and current target raw/NPY bytes; it never copies arrays,
decodes images, changes preparation, or reads label/manifest semantics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import sys


VERSION = 'rgb-conservative-border-prepared-v2'
PREPARED_SIZE = 512
PILLOW = '12.3.0'
HEX = frozenset('0123456789abcdef')
IDENTITY = ('namespace', 'pod', 'uid', 'container', 'python')
ALLOCATION = ('path', 'sha256', 'executor_id', 'threads', 'channel_mask')
FILE_FIELDS = ('path', 'resolved', 'bytes', 'device', 'inode', 'mtime_ns', 'ctime_ns')
MAX_ARRAY_BYTES = PREPARED_SIZE ** 2 * 3 + 16384


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute_path(value):
    text = os.fspath(value)
    require(isinstance(text, str) and '\x00' not in text,
            'an absolute canonical path is required')
    path = Path(text)
    require(path.is_absolute() and str(path) == text and '..' not in path.parts,
            'an absolute canonical path is required: ' + text)
    return path


def sealed_relative(relative, split):
    """Match runtime lexical_path, restricted to the explicitly migrated split."""
    require(isinstance(relative, str) and not any(c in relative for c in ('\\', '\x00', ':')),
            'unsafe relative path')
    path = PurePosixPath(relative)
    require(split in ('train', 'valid') and len(path.parts) == 2
            and path.parts[0] == split and path.as_posix() == relative
            and all(part not in ('.', '..') for part in path.parts),
            'unsafe relative path: ' + relative)
    return path


def full_sha(value):
    require(isinstance(value, str) and len(value) == 64 and set(value) <= HEX,
            'a full lowercase SHA256 is required')
    return value


def policy(data_root):
    return {'version': VERSION, 'prepared_size': PREPARED_SIZE, 'pillow': PILLOW,
            'dataset_root': str(absolute_path(data_root))}


def namespace(value):
    # Default json.dumps separators/ensure_ascii are part of the runtime key.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def entry_key(relative, source_sha):
    return hashlib.sha256((relative + '\0' + full_sha(source_sha)).encode()).hexdigest()


@contextmanager
def directory_fd(path, create=False):
    """Traverse every component with O_NOFOLLOW, including output ancestors."""
    path = absolute_path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptors = [os.open('/', flags)]
    try:
        try:
            for part in path.parts[1:]:
                try:
                    descriptor = os.open(part, flags, dir_fd=descriptors[-1])
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(part, 0o755, dir_fd=descriptors[-1])
                    except FileExistsError:
                        pass
                    descriptor = os.open(part, flags, dir_fd=descriptors[-1])
                descriptors.append(descriptor)
        except OSError as error:
            raise ValueError('unsafe or inaccessible directory route: ' + str(path)) from error
        yield descriptors[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def file_identity(path, info):
    return {'path': str(path), 'resolved': str(path), 'bytes': info.st_size,
            'device': info.st_dev, 'inode': info.st_ino,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def source_file(identity):
    return {'inode': identity['inode'], 'size': identity['bytes'],
            'mtime_ns': identity['mtime_ns'], 'ctime_ns': identity['ctime_ns']}


@contextmanager
def regular_stream(path, maximum=None):
    path = absolute_path(path)
    with directory_fd(path.parent) as parent:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                             | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, 'rb') as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode), 'regular file required: ' + str(path))
            require(maximum is None or before.st_size <= maximum,
                    'oversized file: ' + str(path))
            identity = file_identity(path, before)
            yield stream, identity
            require(identity == file_identity(path, os.fstat(stream.fileno()))
                    and identity == file_identity(path, os.stat(path.name, dir_fd=parent,
                                                               follow_symlinks=False)),
                    'file identity changed during read: ' + str(path))


def content_proof(path, expected_sha=None, maximum=None, keep=False):
    digest, count, blocks = hashlib.sha256(), 0, []
    with regular_stream(path, maximum) as (stream, identity):
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            count += len(block)
            if keep:
                blocks.append(block)
        require(count == identity['bytes'], 'file length changed during read: ' + str(path))
    actual = digest.hexdigest()
    if expected_sha is not None:
        require(actual == full_sha(expected_sha), 'current content SHA256 mismatch: ' + str(path))
    proof = dict(identity, sha256=actual, bytes_hashed=count, evidence='fresh_pod_sha256')
    return (proof, b''.join(blocks)) if keep else proof


def read_json(path, expected_sha=None, maximum=16384):
    proof, raw = content_proof(path, expected_sha, maximum, keep=True)
    value = json.loads(raw)
    require(isinstance(value, dict), 'JSON object required: ' + str(path))
    return value, proof


@contextmanager
def atomic_output(path, replace_progress=False):
    """Publish a fully fsynced temporary file with exclusive hard-link creation."""
    path = absolute_path(path)
    with directory_fd(path.parent, create=True) as parent:
        temporary = '.' + path.name + '.' + secrets.token_hex(12) + '.tmp'
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644, dir_fd=parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                yield stream
                stream.flush()
                os.fsync(stream.fileno())
            if replace_progress:
                # Only the current invocation's exclusively created progress path
                # uses replacement. Cache receipts and other reports never do.
                os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
            else:
                os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent,
                        follow_symlinks=False)
            os.fsync(parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_receipt(path, value):
    raw = json_bytes(value)
    try:
        with atomic_output(path) as stream:
            stream.write(raw)
        created = True
    except FileExistsError:
        _, current = content_proof(path, maximum=16384, keep=True)
        require(current == raw, 'existing target receipt differs; refusing overwrite: ' + str(path))
        created = False
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'created': created}


def validate_copy_row(row, number):
    require(isinstance(row, dict) and isinstance(row.get('asset'), str) and row['asset'],
            f'invalid copy manifest row {number}')
    require(row.get('evidence') == 'fresh_pod_sha256'
            and row.get('source_modified_by_tool') is False and 'error' not in row,
            f'copy content evidence absent or failed at row {number}')
    digest = full_sha(row.get('sha256'))
    require(row.get('target_sha256') == digest and type(row.get('size')) is int
            and row['size'] >= 0, f'copy source/target SHA or size differs at row {number}')
    for side in ('source', 'target'):
        path = str(absolute_path(row.get(side + '_path', '')))
        identity = row.get(side + '_identity')
        require(isinstance(identity, dict) and all(k in identity for k in FILE_FIELDS)
                and identity['path'] == path and identity['resolved'] == path
                and all(type(identity[k]) is int and identity[k] >= 0 for k in FILE_FIELDS[2:])
                and identity['bytes'] == row['size'],
                f'incomplete copy {side} identity at row {number}')


def copy_proofs(manifest_path, manifest_sha, receipt_path, receipt_sha, execution):
    manifest_path, receipt_path = absolute_path(manifest_path), absolute_path(receipt_path)
    value, receipt_content = read_json(receipt_path, full_sha(receipt_sha), maximum=1024 * 1024)
    require(type(value.get('schema_version')) is int and value['schema_version'] == 1
            and value.get('passed') is True, 'copy receipt is not a successful schema-1 proof')
    current = value.get('execution', {})
    require(isinstance(current, dict)
            and all(isinstance(execution.get(k), str) and execution[k]
                    and current.get(k) == execution[k] for k in IDENTITY),
            'copy proof belongs to a different current Pod/interpreter')
    require(isinstance(current.get('cpu_allocation'), dict)
            and isinstance(execution.get('cpu_allocation'), dict)
            and all(k in execution['cpu_allocation']
                    and current['cpu_allocation'].get(k) == execution['cpu_allocation'][k]
                    for k in ALLOCATION), 'copy proof CPU allocation differs')
    require(value.get('manifest') == {'path': str(manifest_path), 'sha256': full_sha(manifest_sha)},
            'copy receipt does not pin the requested manifest path/SHA')
    rows, targets, digest, count = {}, set(), hashlib.sha256(), 0
    with regular_stream(manifest_path) as (stream, identity):
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            count += len(raw)
            row = json.loads(raw)
            validate_copy_row(row, number)
            require(row['source_path'] not in rows and row['target_path'] not in targets,
                    f'duplicate copy source/target at row {number}')
            row['_line'] = number
            rows[row['source_path']] = row
            targets.add(row['target_path'])
        require(count == identity['bytes'] and digest.hexdigest() == manifest_sha,
                'copy manifest current content SHA256 mismatch')
    require(bool(rows), 'empty copy manifest')
    return rows, {'receipt': {'path': str(receipt_path), 'sha256': receipt_content['sha256']},
                  'manifest': {'path': str(manifest_path), 'sha256': digest.hexdigest(),
                               'bytes': count, 'rows': len(rows)}}


def selected_arrays(rows, source_directory, target_directory, expected_count):
    selected = []
    for row in rows.values():
        path = Path(row['source_path'])
        if not path.is_relative_to(source_directory):
            continue
        parts = path.relative_to(source_directory).parts
        require(len(parts) == 2 and len(parts[0]) == 2 and set(parts[0]) <= HEX
                and parts[1].endswith('.npy'), 'unexpected file in copied prepared namespace: ' + str(path))
        key = full_sha(parts[1][:-4])
        require(key[:2] == parts[0], 'prepared key shard differs: ' + str(path))
        target = target_directory / parts[0] / parts[1]
        require(row['target_path'] == str(target), 'copied array target namespace/path differs: ' + str(path))
        selected.append((key, row))
    require(len(selected) == expected_count,
            f'prepared entry count differs: expected {expected_count}, found {len(selected)}')
    require(len({key for key, _ in selected}) == len(selected), 'duplicate prepared target key')
    return sorted(selected, key=lambda item: item[0])


def rebind_entry(item, *, rows, split, source_data_root, target_data_root,
                 old_policy, old_namespace, new_policy):
    """Read/hash and exclusively publish one unique target key; no shared writes."""
    key, array_copy = item
    old_receipt_path = Path(array_copy['source_path']).with_suffix('.json')
    old, old_content = read_json(old_receipt_path)
    require(set(old) == {'policy', 'relative_path', 'source_sha256', 'source_file', 'array_sha256'}
            and old['policy'] == old_policy and namespace(old['policy']) == old_namespace,
            'old prepared receipt policy/fields differ: ' + str(old_receipt_path))
    relative = str(sealed_relative(old['relative_path'], split))
    source_sha, array_sha = full_sha(old['source_sha256']), full_sha(old['array_sha256'])
    require(entry_key(relative, source_sha) == key,
            'old prepared receipt key/namespace differs: ' + str(old_receipt_path))
    old_raw, new_raw = source_data_root / relative, target_data_root / relative
    raw_copy = rows.get(str(old_raw))
    require(raw_copy is not None and raw_copy['target_path'] == str(new_raw)
            and raw_copy['sha256'] == source_sha,
            'raw copy proof missing or sealed path/SHA differs: ' + relative)
    require(array_copy['sha256'] == array_sha, 'array copy proof SHA differs: ' + relative)
    expected_source_file = source_file(raw_copy['source_identity'])
    require(isinstance(old['source_file'], dict)
            and set(old['source_file']) == set(expected_source_file)
            and all(type(v) is int for v in old['source_file'].values())
            and old['source_file'] == expected_source_file,
            'old receipt source_file differs from pinned copy source identity: '
            + relative + '; receipt=' + repr(old['source_file'])
            + '; copy_source=' + repr(expected_source_file))
    raw_current = content_proof(new_raw, source_sha)
    array_current = content_proof(array_copy['target_path'], array_sha, MAX_ARRAY_BYTES)
    require(raw_current['bytes'] == raw_copy['size'] and array_current['bytes'] == array_copy['size'],
            'current target sizes differ from pinned copy proof: ' + relative)
    new_receipt = dict(old, policy=new_policy, source_file=source_file(raw_current))
    new_receipt_path = Path(array_copy['target_path']).with_suffix('.json')
    receipt = write_receipt(new_receipt_path, new_receipt)
    return {'split': split, 'key': key, 'relative_path': relative,
            'source_sha256': source_sha, 'array_sha256': array_sha,
            'source_receipt': {'path': str(old_receipt_path), 'sha256': old_content['sha256']},
            'target_receipt': receipt, 'source_file': new_receipt['source_file'],
            'copy_manifest_lines': {'raw': raw_copy['_line'], 'array': array_copy['_line']},
            'raw': {'source_path': str(old_raw), 'target_path': str(new_raw),
                    'sha256': raw_current['sha256'], 'bytes': raw_current['bytes']},
            'array': {'source_path': array_copy['source_path'], 'target_path': array_copy['target_path'],
                      'sha256': array_current['sha256'], 'bytes': array_current['bytes']}}


def rebind_cache(*, source_cache_root, target_cache_root, source_data_root, target_data_root,
                 split, expected_count, copy_manifest, copy_manifest_sha256,
                 copy_receipt, copy_receipt_sha256, output, execution):
    require(split in ('train', 'valid'), 'only train/valid prepared entries may be rebound')
    require(type(expected_count) is int and expected_count > 0, 'explicit positive expected_count required')
    threads = execution.get('cpu_allocation', {}).get('threads')
    require(type(threads) is int and threads > 0, 'explicit positive CPU thread budget required')
    # A one-thread fixture/channel stays inline; main plus a worker would exceed it.
    workers = max(1, threads - 1) if threads > 1 else 0
    roots = [absolute_path(p) for p in (source_cache_root, target_cache_root,
                                      source_data_root, target_data_root)]
    source_cache_root, target_cache_root, source_data_root, target_data_root = roots
    # Rebinding must never publish inside a source data/cache tree.
    require(not any(target.is_relative_to(source) or source.is_relative_to(target)
                    for source in (source_cache_root, source_data_root)
                    for target in (target_cache_root, target_data_root)),
            'source and target roots must be disjoint')
    for root in roots:
        with directory_fd(root):
            pass
    old_policy, new_policy = policy(source_data_root), policy(target_data_root)
    old_namespace, new_namespace = namespace(old_policy), namespace(new_policy)
    source_directory = source_cache_root / split / 'sealed' / old_namespace
    target_directory = target_cache_root / split / 'sealed' / new_namespace
    rows, copy_evidence = copy_proofs(copy_manifest, copy_manifest_sha256,
                                      copy_receipt, copy_receipt_sha256, execution)
    arrays = selected_arrays(rows, source_directory, target_directory, expected_count)
    output = absolute_path(output)
    entries_path = output.with_suffix('.entries.jsonl')
    started_path, progress_path = output.with_suffix('.started.json'), output.with_suffix('.progress.json')
    output_paths = (output, entries_path, started_path, progress_path)
    require(len(set(output_paths)) == len(output_paths), 'migration report paths must differ')
    require(not any(path.is_relative_to(root) for root in roots for path in output_paths),
            'migration reports must stay outside source/target asset trees')
    for path in output_paths:
        with directory_fd(path.parent, create=True) as parent:
            try:
                os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ValueError('refusing to overwrite existing output: ' + str(path))
    totals = {'entries': 0, 'raw_bytes': 0, 'array_bytes': 0,
              'receipts_created': 0, 'receipts_reused': 0}
    start = datetime.now(timezone.utc).isoformat()
    implementation_sha = content_proof(Path(__file__).absolute())['sha256']
    # Bounded map batches avoid queuing all 57245 entries on Python 3.12. Each
    # batch preserves the existing sorted-key order, as does pool.map itself.
    batch_size = max(1, workers * 2)
    parallelism = {'allocated_threads': threads, 'workers': workers, 'main_threads': 1,
                   'max_process_threads': workers + 1, 'max_in_flight_entries': batch_size,
                   'mode': 'thread-pool' if workers else 'inline', 'result_order': 'sorted-key'}
    with atomic_output(started_path) as stream:
        stream.write(json_bytes({'schema_version': 1, 'scope': 'start_only', 'status': 'started',
            'passed': False, 'start': start, 'at': start, 'split': split, 'expected_count': expected_count,
            'output': str(output), 'execution': execution, 'parallelism': parallelism,
            'script_sha256': implementation_sha}))

    def progress(status, *, first=False, error=None):
        value = {'schema_version': 1, 'scope': 'progress_only', 'status': status,
                 'start': start, 'at': datetime.now(timezone.utc).isoformat(),
                 'split': split, 'expected_count': expected_count, 'output': str(output),
                 'entries': totals['entries'], 'raw_bytes': totals['raw_bytes'],
                 'array_bytes': totals['array_bytes'], 'parallelism': parallelism}
        if error is not None:
            value.update(error_type=type(error).__name__, error=str(error))
        with atomic_output(progress_path, replace_progress=not first) as stream:
            stream.write(json_bytes(value))

    progress('running', first=True)
    process_entry = partial(rebind_entry, rows=rows, split=split,
                            source_data_root=source_data_root, target_data_root=target_data_root,
                            old_policy=old_policy, old_namespace=old_namespace, new_policy=new_policy)
    entries_sha, entries_bytes = hashlib.sha256(), 0
    try:
        pool_context = ThreadPoolExecutor(max_workers=workers) if workers else nullcontext()
        with pool_context as pool, atomic_output(entries_path) as stream:
            for offset in range(0, len(arrays), batch_size):
                batch = arrays[offset:offset + batch_size]
                results = pool.map(process_entry, batch) if pool else map(process_entry, batch)
                for entry in results:
                    encoded = json_bytes(entry)
                    stream.write(encoded)
                    entries_sha.update(encoded)
                    entries_bytes += len(encoded)
                    totals['entries'] += 1
                    totals['raw_bytes'] += entry['raw']['bytes']
                    totals['array_bytes'] += entry['array']['bytes']
                    totals['receipts_created' if entry['target_receipt']['created'] else 'receipts_reused'] += 1
                    if totals['entries'] % 512 == 0:
                        progress('running')
    except Exception as error:
        # The executor has joined its bounded in-flight work before reporting a
        # failure; no worker, lease, timer or background observer is left behind.
        try:
            progress('failed', error=error)
        except (OSError, ValueError) as progress_error:
            print(json.dumps({'scope': 'progress_only', 'status': 'unknown',
                              'error': str(progress_error)}), file=sys.stderr)
        raise
    progress('entries-processed')
    report = {'schema_version': 1, 'passed': True, 'stage': 'prepared-cache-rebind',
              'start': start, 'at': datetime.now(timezone.utc).isoformat(),
              'execution': execution, 'split': split, 'parallelism': parallelism,
              'expected_count': expected_count, 'counts': totals,
              'source': {'cache_root': str(source_cache_root), 'policy': old_policy, 'namespace': old_namespace},
              'target': {'cache_root': str(target_cache_root), 'policy': new_policy, 'namespace': new_namespace},
              'copy_proof': copy_evidence,
              'entries_manifest': {'path': str(entries_path), 'sha256': entries_sha.hexdigest(), 'bytes': entries_bytes},
              'byte_evidence': {'source_raw_and_arrays': 'pinned-copy-content-proof',
                                'target_raw_and_arrays': 'fresh_pod_sha256',
                                'target_bytes_hashed': totals['raw_bytes'] + totals['array_bytes'],
                                'old_raw_or_array_bytes_reread': 0,
                                'preparation_changed': False, 'arrays_copied_by_rebind': False,
                                'old_receipts_modified': False, 'labels_or_manifest_semantics_read': False},
              'stage_reports': {'started': str(started_path), 'progress': str(progress_path)},
              'script_sha256': implementation_sha}
    with atomic_output(output) as stream:
        stream.write(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-cache-root', 'target-cache-root', 'source-data-root', 'target-data-root',
                 'copy-manifest', 'copy-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--copy-manifest-sha256', required=True)
    parser.add_argument('--copy-receipt-sha256', required=True)
    parser.add_argument('--split', choices=('train', 'valid'), required=True)
    parser.add_argument('--expected-count', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretraining'))
    from source_chain import cpu_context
    try:
        report = rebind_cache(**vars(args), execution=cpu_context())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({'passed': False, 'error': str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({'passed': True, 'receipt': str(args.output),
                      'receipt_sha256': content_proof(args.output)['sha256'],
                      'counts': report['counts'], 'namespace': report['target']['namespace']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
