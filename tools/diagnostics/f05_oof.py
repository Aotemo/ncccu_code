"""Construct and validate fixed fit-only folds for the F05 OOF program."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def build_fold_manifests(base, seed):
    """Partition only the original fit groups, keeping the original dev fixed."""
    from retrain.data import _partition_groups
    if type(seed) is not int or not isinstance(base, dict) or not isinstance(base.get('rows'), list):
        raise ValueError('explicit base manifest and split seed required')
    rows = base['rows']; fit = base['fit_indices']; dev = base['dev_indices']
    if (not fit or not dev or base.get('audit_indices') not in (None, [])
            or len(set(fit)) != len(fit) or len(set(dev)) != len(dev) or set(fit) & set(dev)
            or any(type(i) is not int or not 0 <= i < len(rows) for i in [*fit, *dev])
            or set(fit) | set(dev) != set(range(len(rows)))
            or any(row.get('label') not in (0, 1) or type(row.get('label')) is not int
                   or not isinstance(row.get('group'), str) or not row['group'] for row in rows)):
        raise ValueError('base fit/dev population or labels differ')
    if {rows[i]['group'] for i in fit} & {rows[i]['group'] for i in dev}:
        raise ValueError('original content group crosses fit and dev')
    parts = _partition_groups(rows, fit, [1 / 5] * 5, seed)
    if len(parts) != 5 or sorted(i for part in parts for i in part) != sorted(fit):
        raise ValueError('fit groups are not covered exactly once')
    result = []
    for number, held in enumerate(parts):
        held_set = set(held)
        train = sorted(i for i in fit if i not in held_set)
        if (not train or not held or {rows[i]['label'] for i in train} != {0, 1}
                or {rows[i]['label'] for i in held} != {0, 1}
                or {rows[i]['group'] for i in train} & {rows[i]['group'] for i in held}):
            raise ValueError('fold training/held-out classes or groups differ')
        fold = dict(base)
        fold.update(fit_indices=train, dev_indices=list(dev), audit_indices=list(held),
                    split={'kind': 'fit_only_5fold', 'seed': seed, 'fold': number, 'folds': 5,
                           'base_split': base.get('split')},
                    split_method='content-group-label-age-size-v1')
        fold['counts'] = {name: {'rows': len(indices), 'positive': sum(rows[i]['label'] for i in indices),
                                 'groups': len({rows[i]['group'] for i in indices})}
                          for name, indices in (('fit', train), ('dev', dev), ('audit', held))}
        result.append(fold)
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    args = parser.parse_args()
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(key) != '' for key in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))):
        raise ValueError('hidden-device Pod preparation required')
    source = Path(__file__).resolve().parents[2]
    config = json.loads((source / 'configs/F05-FOLDS-5.json').read_text())
    if args.base_manifest.is_symlink() or sha(args.base_manifest) != config['base_manifest_sha256']:
        raise ValueError('sealed original fit/dev manifest differs')
    if (not args.output.is_absolute() or args.output.exists()
            or not args.output.parent.resolve(strict=True).is_relative_to(args.work_root.resolve() / 'technical-prep')):
        raise ValueError('fresh isolated fold output required')
    sys.path.insert(0, str(source / 'runtime/bundle'))
    from retrain.age import make_spec
    from retrain.persistence import canonical_digest
    base = json.loads(args.base_manifest.read_text())
    if (len(base['fit_indices']) != config['fit_count'] or len(base['dev_indices']) != config['dev_count']
            or config['folds'] != 5 or config['official_read'] is not False):
        raise ValueError('predeclared population differs')
    folds = build_fold_manifests(base, config['fold_seed'])
    args.output.mkdir()
    rows = []
    for number, fold in enumerate(folds):
        age = make_spec(fold)
        rows.append({'fold': number, 'round': config['rounds'][number], 'counts': fold['counts'],
                     'fit_indices': fold['fit_indices'], 'audit_indices': fold['audit_indices'],
                     'fit_indices_sha256': canonical_digest(fold['fit_indices']),
                     'audit_indices_sha256': canonical_digest(fold['audit_indices']),
                     'dev_indices_sha256': canonical_digest(fold['dev_indices']),
                     'manifest_canonical_sha256': canonical_digest(fold), 'age': age})
    result = {'status': 'five_fold_fit_only_indices_sealed_no_training', 'at': time.time(),
              'base_manifest_sha256': config['base_manifest_sha256'], 'seed': config['fold_seed'], 'folds': rows,
              'GPU_initialized': False, 'training_started': False, 'official_read': False}
    output = args.output / 'FOLDS-private.json'
    with output.open('x') as stream:
        json.dump(result, stream, separators=(',', ':'), allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'status': result['status'], 'fold_counts': [row['counts'] for row in rows],
                      'receipt_sha256': sha(output)}), flush=True)


if __name__ == '__main__':
    main()
