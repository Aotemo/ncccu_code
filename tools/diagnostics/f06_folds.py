"""Register F06 three-fold splits and check the official prepared cache (CPU only).

Fold assignment is label-stratified with seed 20261106. When the official CSV
carries patient or eye identifiers, whole groups are assigned together.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

NAME_FIELDS = ('name', 'image_name', 'image', 'filename')
LABEL_FIELDS = ('true_label', 'label', 'target', 'y')
GROUP_HINTS = ('patient', 'subject', 'case', 'eye', 'laterality', 'side')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pick(fieldnames, wanted, contains=None):
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in wanted:
        if candidate in lowered:
            return lowered[candidate]
    if contains is not None:
        matches = [name for lower, name in lowered.items() if contains in lower]
        if len(matches) == 1:
            return matches[0]
    return None


def read_labels(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    name_field = pick(fieldnames, NAME_FIELDS, 'name')
    label_field = pick(fieldnames, LABEL_FIELDS, 'label')
    if name_field is None or label_field is None:
        raise ValueError('name and label columns required')
    groups = [name for name in fieldnames if any(hint in name.lower() for hint in GROUP_HINTS)]
    nulls = {}
    for field in fieldnames:
        nulls[field] = sum(1 for row in rows if (row.get(field) or '').strip() == '')
    names, labels = [], []
    for row in rows:
        name = (row.get(name_field) or '').strip()
        if not name:
            raise ValueError('empty image name')
        try:
            label = int((row.get(label_field) or '').strip())
        except (TypeError, ValueError) as error:
            raise ValueError('integer label required') from error
        if label not in (0, 1):
            raise ValueError('binary label required')
        names.append(name)
        labels.append(label)
    if len(set(names)) != len(names):
        raise ValueError('duplicate image names')
    return {'fieldnames': fieldnames, 'rows': rows, 'names': names, 'labels': labels,
            'name_field': name_field, 'label_field': label_field, 'group_fields': groups,
            'nulls': nulls}


def group_keys(table):
    if not table['group_fields']:
        return [name for name in table['names']]
    keys = []
    for index, row in enumerate(table['rows']):
        parts = [(row.get(field) or '').strip() for field in table['group_fields']]
        keys.append('|'.join(parts) if any(parts) else table['names'][index])
    return keys


def assign(table, seed, folds=3):
    """Exact stratified split per image, or whole-group greedy split when grouped."""
    keys = group_keys(table)
    labels = table['labels']
    rng = random.Random(seed)
    if not table['group_fields']:
        folds_out = [0] * len(keys)
        for label in (0, 1):
            indices = [index for index, value in enumerate(labels) if value == label]
            rng.shuffle(indices)
            base, extra = divmod(len(indices), folds)
            start = 0
            for fold in range(folds):
                take = base + (1 if fold < extra else 0)
                for index in indices[start:start + take]:
                    folds_out[index] = fold
                start += take
        return folds_out, len(set(keys))
    members = {}
    for index, key in enumerate(keys):
        members.setdefault(key, []).append(index)
    if any(len({labels[i] for i in indices}) != 1 for indices in members.values()):
        raise ValueError('a group carries both labels; grouped stratification is impossible')
    positives_total = sum(1 for value in labels if value == 1)
    negatives_total = len(labels) - positives_total
    base, extra = divmod(positives_total, folds)
    positive_target = [base + (1 if index < extra else 0) for index in range(folds)]
    base, extra = divmod(negatives_total, folds)
    negative_target = [base + (1 if index < extra else 0) for index in range(folds)]
    positives_have, negatives_have = [0] * folds, [0] * folds
    groups = list(members)
    rng.shuffle(groups)
    groups.sort(key=lambda key: -len(members[key]))
    fold_of_key = {}
    for key in groups:
        size = len(members[key])
        label = labels[members[key][0]]
        if label == 1:
            score = [(positives_have[f] + size - positive_target[f], positives_have[f], f) for f in range(folds)]
        else:
            score = [(negatives_have[f] + size - negative_target[f], negatives_have[f], f) for f in range(folds)]
        fold = min(score)[2]
        fold_of_key[key] = fold
        if label == 1:
            positives_have[fold] += size
        else:
            negatives_have[fold] += size
    return [fold_of_key[key] for key in keys], len(members)


def counts(labels, folds_out, folds=3):
    summary = []
    for fold in range(folds):
        indices = [index for index, value in enumerate(folds_out) if value == fold]
        positive = sum(labels[index] for index in indices)
        summary.append({'fold': fold, 'rows': len(indices), 'positive': positive,
                        'negative': len(indices) - positive})
    return summary


def build(labels_path, seed=20261106):
    table = read_labels(labels_path)
    folds_out, group_count = assign(table, seed)
    summary = counts(table['labels'], folds_out)
    positives_total = sum(item['positive'] for item in summary)
    negatives_total = sum(item['negative'] for item in summary)
    if sum(item['rows'] for item in summary) != len(table['names']):
        raise ValueError('fold rows do not cover the official population')
    return {'format': 'f06-official-folds-v1', 'seed': seed,
            'method': ('group-stratified by ' + '+'.join(table['group_fields'])
                       if table['group_fields'] else 'label-stratified per image'),
            'group_fields': table['group_fields'], 'group_count': group_count,
            'population': {'rows': len(table['names']), 'positive': positives_total,
                           'negative': negatives_total},
            'folds': summary,
            'columns': {'all': table['fieldnames'], 'name': table['name_field'],
                        'label': table['label_field'], 'null_counts': table['nulls']},
            'rows': [{'name': name, 'label': label, 'fold': fold}
                     for name, label, fold in zip(table['names'], table['labels'], folds_out)],
            'official_read': True, 'test_labels_read': False, 'forward_calls': 0,
            'GPU_initialized': False}


def write(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError('fresh output required')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=1, allow_nan=False).encode('utf-8') + b'\n'
    with path.open('xb') as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def inspect(path):
    table = read_labels(path)
    return {'path': str(path), 'sha256': sha256(path), 'rows': len(table['names']),
            'columns': table['fieldnames'], 'name_column': table['name_field'],
            'label_column': table['label_field'], 'group_columns': table['group_fields'],
            'null_counts': table['nulls'],
            'positive': sum(table['labels']), 'negative': len(table['labels']) - sum(table['labels'])}


def cache_check(path, data_root, cache_root, prepared_size, limit=None):
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'))
    from retrain.preprocessing import PreparedImageCache
    table = read_labels(path)
    sha_field = pick(table['fieldnames'], ('sha256', 'sha', 'image_sha256'), 'sha')
    cache = PreparedImageCache(Path(data_root), Path(cache_root), int(prepared_size))
    missing, present, mismatch = [], 0, []
    rows = table['rows'][:limit] if limit else table['rows']
    for row, name in zip(rows, table['names']):
        relative = 'valid/' + name if not name.startswith('valid/') else name
        expected = (row.get(sha_field) or '').strip() if sha_field else ''
        if len(expected) != 64:
            source = Path(data_root) / relative
            if not source.is_file():
                missing.append(relative)
                continue
            expected = sha256(source)
        try:
            cache.read(relative, expected)
            present += 1
        except (ValueError, OSError, KeyError) as error:
            mismatch.append({'relative': relative, 'error': type(error).__name__ + ': ' + str(error)[:160]})
    return {'status': 'official_prepared_cache_coverage', 'labels': {'path': str(path), 'sha256': sha256(path)},
            'data_root': str(data_root), 'cache_root': str(cache_root), 'prepared_size': prepared_size,
            'cache_namespace': cache.directory.name, 'rows': len(rows), 'present': present,
            'missing': len(missing), 'mismatch': len(mismatch),
            'coverage': present / len(rows) if rows else 0.0,
            'missing_sample': missing[:5], 'mismatch_sample': mismatch[:5],
            'official_read': True, 'test_labels_read': False, 'forward_calls': 0,
            'GPU_initialized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'folds', 'cache'))
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--prepared-size', type=int, default=512)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'inspect':
        print(json.dumps(inspect(args.labels), indent=1, sort_keys=True))
        return
    if args.command == 'folds':
        if args.output is None or args.receipt is None:
            raise ValueError('folds requires --output and --receipt')
        value = build(args.labels, args.seed)
        digest, size = write(args.output, value)
        receipt = {'status': 'f06_official_folds_registered', 'folds': str(args.output),
                   'folds_sha256': digest, 'bytes': size,
                   'labels': {'path': str(args.labels), 'sha256': sha256(args.labels)},
                   'seed': args.seed, 'method': value['method'],
                   'population': value['population'], 'fold_counts': value['folds'],
                   'group_fields': value['group_fields'],
                   'official_read': True, 'test_labels_read': False, 'TTA': False}
        digest2, _ = write(args.receipt, receipt)
        print(json.dumps({'folds': str(args.output), 'folds_sha256': digest,
                          'receipt_sha256': digest2, **value['population'],
                          'fold_counts': value['folds']}, sort_keys=True))
        return
    if args.data_root is None or args.cache_root is None or args.output is None:
        raise ValueError('cache requires --data-root, --cache-root and --output')
    value = cache_check(args.labels, args.data_root, args.cache_root, args.prepared_size, args.limit)
    digest, _ = write(args.output, value)
    print(json.dumps({'output': str(args.output), 'sha256': digest, 'coverage': value['coverage'],
                      'present': value['present'], 'missing': value['missing'],
                      'mismatch': value['mismatch'], 'namespace': value['cache_namespace']}, sort_keys=True))


if __name__ == '__main__':
    main()
