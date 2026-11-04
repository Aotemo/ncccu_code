"""Read-only, rate-limited, resumable inventory of possible model files.

Run through the verified Pod CPU channel.  The private inventory is evidence for
later classification, not permission to delete any file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def inside(path, parent):
    try:
        Path(path).relative_to(Path(parent))
        return True
    except ValueError:
        return False


def potential_weight(name, parent):
    name = name.lower()
    parent_parts = [part.lower() for part in Path(parent).parts]
    if name.endswith(('.pt', '.pth', '.pt2', '.ckpt', '.safetensors', '.bin',
                      '.pkl', '.pickle', '.h5', '.hdf5', '.onnx', '.weights', '.npz', '.pb')):
        return True
    relevant = any(word in name for word in ('model', 'weight', 'checkpoint', 'tensor', 'state'))
    model_dir = any(word in part for part in parent_parts for word in ('checkpoint', 'weight', 'model'))
    if name.endswith('.npy'):
        protected_image_cache = any(part in ('cache', 'data') or part.startswith(('prepared', 'image-cache'))
                                    for part in parent_parts)
        return relevant or model_dir or not protected_image_cache
    if name.endswith(('.tar', '.tar.gz', '.tgz', '.zip')):
        return relevant
    return False


def new_scan(root, excluded):
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('a real existing inventory root is required')
    excluded = [str(Path(path).resolve()) for path in excluded]
    if any(not inside(path, root) for path in excluded):
        raise ValueError('excluded output must be within the inventory root')
    return {'schema': 1, 'root': str(root), 'excluded': excluded,
            'stack': [{'path': str(root), 'names': None, 'offset': 0}],
            'files': [], 'symlinks': [], 'errors': [], 'unstable_directories': [],
            'visited_entries': 0, 'complete': False, 'created_at': time.time()}


def scan_chunk(state, *, max_entries, delay, seconds):
    if state['complete']:
        return state
    if max_entries < 1 or delay < 0 or seconds <= 0:
        raise ValueError('positive bounded scan limits are required')
    root = Path(state['root'])
    start = time.monotonic()
    processed = 0
    while state['stack'] and processed < max_entries and time.monotonic() - start < seconds:
        frame = state['stack'][-1]
        directory = Path(frame['path'])
        if frame['names'] is None:
            try:
                if not inside(directory, root) or directory.resolve(strict=True) != directory:
                    raise ValueError('directory changed or resolves outside root')
                before = directory.stat()
                frame['mtime_ns'] = before.st_mtime_ns
                names = os.listdir(directory)
                if directory == root:
                    # Inspect round checkpoints and preparation copies first; all
                    # other roots remain in the same complete, resumable scan.
                    priority = {'rounds': 0, 'technical-prep': 1}
                    frame['names'] = sorted(names, key=lambda name: (priority.get(name, 2), name))
                else:
                    frame['names'] = sorted(names)
            except (OSError, ValueError) as error:
                state['errors'].append({'path': str(directory), 'reason': repr(error)})
                state['stack'].pop()
                continue
        if frame['offset'] >= len(frame['names']):
            try:
                if directory.stat().st_mtime_ns != frame['mtime_ns']:
                    state['unstable_directories'].append(str(directory))
            except OSError as error:
                state['errors'].append({'path': str(directory), 'reason': repr(error)})
            state['stack'].pop()
            continue
        name = frame['names'][frame['offset']]
        frame['offset'] += 1
        path = directory / name
        processed += 1
        state['visited_entries'] += 1
        if delay:
            time.sleep(delay)
        if any(inside(path, excluded) for excluded in state['excluded']):
            continue
        try:
            value = path.lstat()
            if stat.S_ISLNK(value.st_mode):
                state['symlinks'].append(str(path))
            elif stat.S_ISDIR(value.st_mode):
                state['stack'].append({'path': str(path), 'names': None, 'offset': 0})
            elif stat.S_ISREG(value.st_mode) and potential_weight(name, directory):
                state['files'].append({'path': str(path), 'relative_path': str(path.relative_to(root)),
                    'device': value.st_dev, 'inode': value.st_ino, 'size': value.st_size,
                    'allocated': value.st_blocks * 512, 'mtime_ns': value.st_mtime_ns,
                    'mode': value.st_mode, 'nlink': value.st_nlink})
        except OSError as error:
            state['errors'].append({'path': str(path), 'reason': repr(error)})
    state['complete'] = not state['stack']
    state['last_chunk_at'] = time.time()
    state['last_chunk_entries'] = processed
    return state


def write_state(path, value):
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--max-entries', type=int, default=2500)
    parser.add_argument('--seconds', type=float, default=35)
    parser.add_argument('--delay', type=float, default=.01)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    target = args.state.absolute()
    if not inside(target, root / 'technical-prep') or not target.parent.is_dir():
        raise ValueError('inventory state must be in an existing isolated technical-prep directory')
    if args.delay < .01 or not 1 <= args.seconds <= 45 or not 1 <= args.max_entries <= 3000:
        raise ValueError('explicit bounded NFS rate and chunk limits required')
    if any(os.environ.get(key) != '' for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')):
        raise ValueError('GPU visibility must be hidden')
    if target.exists():
        state = json.loads(target.read_text())
        if state['root'] != str(root) or state['excluded'] != [str(target.parent.resolve())]:
            raise ValueError('resume source or protected output changed')
    else:
        state = new_scan(root, [target.parent])
    state = scan_chunk(state, max_entries=args.max_entries, delay=args.delay, seconds=args.seconds)
    write_state(target, state)
    groups = {}
    for item in state['files']:
        group = item['relative_path'].split('/')[0]
        value = groups.setdefault(group, {'count': 0, 'logical_bytes': 0, 'allocated_bytes': 0})
        value['count'] += 1
        value['logical_bytes'] += item['size']
        value['allocated_bytes'] += item['allocated']
    print(json.dumps({'status': 'complete' if state['complete'] else 'partial_resumable',
        'files_found': len(state['files']), 'visited_entries': state['visited_entries'],
        'last_chunk_entries': state['last_chunk_entries'], 'pending_directories': len(state['stack']),
        'errors': len(state['errors']), 'unstable_directories': len(state['unstable_directories']),
        'symlinks_not_followed': len(state['symlinks']), 'groups': groups,
        'state_path': str(target), 'state_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'checkpoint_loaded': False, 'files_deleted': False, 'official_read': False}, indent=2), flush=True)


if __name__ == '__main__':
    main()
