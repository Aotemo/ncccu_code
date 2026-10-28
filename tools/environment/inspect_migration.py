"""Resolve only inventoried training dependencies before planning their relocation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--train-identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.inventory) != args.inventory_sha256:
        raise ValueError('inventory pin differs')
    inventory = json.loads(args.inventory.read_text())
    if inventory['uid'] != os.environ['NCCCU_POD_UID'] or inventory['exit_code'] != 0:
        raise ValueError('successful current Pod inventory required')
    files = args.inventory.parent / 'FILES.jsonl'
    if sha(files) != inventory['file_inventory_sha256']:
        raise ValueError('file inventory pin differs')
    assets = {value['id']: value for value in inventory['assets']}
    cache_root = Path(assets['prepared512_pillow123']['path'])
    counts, sizes, samples = Counter(), Counter(), {}
    with files.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['asset'] != 'prepared512_pillow123' or row['kind'] != 'file':
                continue
            relative = Path(row['path']).relative_to(cache_root)
            group = '/'.join(relative.parts[:3])
            counts[group] += 1; sizes[group] += row['bytes']
            if relative.suffix == '.json' and group not in samples:
                value = json.loads(Path(row['path']).read_text())
                samples[group] = {'path': row['path'], 'keys': sorted(value),
                    'policy': value.get('policy'), 'source_file': value.get('source_file'),
                    'relative_path': value.get('relative_path')}
    root = Path(assets['fixed_9010_stage1_18_objects']['path'])
    freeze = json.loads((root / 'FREEZE.json').read_text())
    if sha(root / 'FREEZE.json') != '90e063dca3ab22225308043ba4b0de379688675ef00606d00b8b36e2135bc0c6':
        raise ValueError('stage1 FREEZE differs')
    checks = {}
    for name, record in freeze['files'].items():
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('frozen metadata path escapes source')
        actual = sha(path)
        if actual != record['sha256']:
            raise ValueError('stage1 frozen bytes changed: ' + name)
        checks[name] = {'bytes': path.stat().st_size, 'sha256': actual}
    identity_path = args.train_identity
    if not identity_path.is_absolute() or identity_path.is_symlink():
        raise ValueError('explicit absolute training identity required')
    result = {'scope': 'migration_dependencies_only_no_copy_no_official_semantics',
        'uid': os.environ['NCCCU_POD_UID'], 'inventory_sha256': args.inventory_sha256,
        'cache_groups': {key: {'files': counts[key], 'bytes': sizes[key], 'sample': samples.get(key)} for key in sorted(counts)},
        'stage1': {'root': str(root), 'frozen_files': checks, 'frozen_count': len(checks),
                   'total_with_FREEZE': len(checks) + 1},
        'train_identity': {'path': str(identity_path), 'bytes': identity_path.stat().st_size,
                           'sha256': sha(identity_path)},
        'copies_started': False, 'source_modified': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'receipt': str(args.output), 'sha256': sha(args.output), **result}, indent=2))


if __name__ == '__main__':
    main()
