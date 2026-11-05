"""Fuse sealed member test scores into a submittable CSV (CPU, no official labels).

Fixed rule: standardise every member by its internal-dev mean and population
standard deviation, average the standardised scores with the manifest weights,
fit a deterministic one-dimensional Platt map on the fused internal-dev score,
and apply that map to the fused test score.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TEST_ROWS = 5855
NAME_FIELDS = ('image_name', 'name')
PROBABILITY_FIELDS = ('probability', 'score', 'logit')
SPEARMAN_REFERENCE = 'R120'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _field(fieldnames, wanted):
    lowered = {name.lower(): name for name in fieldnames or []}
    for candidate in wanted:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_scores(path, *, need_label=False):
    """Read a name/probability CSV, keeping file order and rejecting duplicates."""
    path = Path(path)
    names, values, labels = [], [], []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        name_field = _field(reader.fieldnames, NAME_FIELDS)
        value_field = _field(reader.fieldnames, PROBABILITY_FIELDS)
        label_field = _field(reader.fieldnames, ('label',))
        if name_field is None or value_field is None or (need_label and label_field is None):
            raise ValueError('name, probability and label columns required in ' + str(path))
        for row in reader:
            name = (row[name_field] or '').strip()
            if not name:
                raise ValueError('empty image name in ' + str(path))
            try:
                value = float(row[value_field])
            except (TypeError, ValueError) as error:
                raise ValueError('numeric probability required in ' + str(path)) from error
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('probability outside [0,1] in ' + str(path))
            names.append(name)
            values.append(value)
            if need_label:
                try:
                    label = int(row[label_field])
                except (TypeError, ValueError) as error:
                    raise ValueError('binary dev label required in ' + str(path)) from error
                if label not in (0, 1):
                    raise ValueError('binary dev label required in ' + str(path))
                labels.append(label)
    if not names or len(set(names)) != len(names):
        raise ValueError('unique image names required in ' + str(path))
    return {'path': str(path), 'sha256': sha(path), 'names': names,
            'values': np.asarray(values, dtype=np.float64),
            'labels': np.asarray(labels, dtype=np.int64) if need_label else None}


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean(dtype=np.float64)), float(values.std(dtype=np.float64, ddof=0))


def standardise(values, mean, std):
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError('positive finite dev moments required')
    return (np.asarray(values, dtype=np.float64) - mean) / std


def fuse(vectors, weights):
    """Weighted mean of distinct aligned standardised vectors."""
    names = list(vectors)
    if not names or len(weights) != len(names):
        raise ValueError('member vectors and weights must correspond')
    total = float(sum(weights))
    if not math.isfinite(total) or total <= 0:
        raise ValueError('positive finite weight sum required')
    stacked = np.stack([np.asarray(vectors[name], dtype=np.float64) for name in names])
    if not np.isfinite(stacked).all():
        raise ValueError('finite member vectors required')
    return np.average(stacked, axis=0, weights=np.asarray(weights, dtype=np.float64))


def fit_platt(scores, labels):
    """Deterministic one-dimensional logistic fit; returns slope/intercept."""
    from sklearn.linear_model import LogisticRegression
    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels)
    if scores.shape[0] != labels.shape[0] or set(labels.tolist()) != {0, 1}:
        raise ValueError('two-class aligned dev scores required')
    if not np.isfinite(scores).all():
        raise ValueError('finite dev scores required')
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, tol=1e-10, random_state=20261105)
    model.fit(scores, labels)
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        raise ValueError('positive finite Platt parameters required')
    return {'format': 'platt-one-dimensional-v1', 'slope': slope, 'intercept': intercept,
            'config': {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 1e-10,
                       'random_state': 20261105, 'feature': 'fused_dev_score'},
            'dev_count': int(scores.shape[0]), 'dev_positive': int(labels.sum())}


def apply_platt(scores, parameters):
    values = parameters['slope'] * np.asarray(scores, dtype=np.float64) + parameters['intercept']
    values = 1.0 / (1.0 + np.exp(-values))
    if not np.isfinite(values).all():
        raise ValueError('finite Platt probabilities required')
    return np.clip(values, 0.0, 1.0)


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman(left, right):
    a, b = average_ranks(left), average_ranks(right)
    a_centred, b_centred = a - a.mean(), b - b.mean()
    scale = math.sqrt(float((a_centred ** 2).sum()) * float((b_centred ** 2).sum()))
    if scale == 0:
        raise ValueError('constant ranking cannot be correlated')
    return float((a_centred * b_centred).sum() / scale)


def read_members(manifest, *, allow_fixture=False):
    if manifest.get('format') != 'f05-fusion-members-v1' or manifest.get('official_read') is not False:
        raise ValueError('f05-fusion-members-v1 manifest with official_read=false required')
    rows = manifest.get('members')
    if not isinstance(rows, list) or not rows:
        raise ValueError('at least one member required')
    identifiers = [row.get('id') for row in rows]
    if any(not isinstance(name, str) or not name for name in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError('unique string member ids required')
    members = []
    for row in rows:
        for key in ('test', 'dev'):
            reference = row.get(key)
            if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
                raise ValueError('exact {path, sha256} reference required for ' + key)
            if sha(reference['path']) != reference['sha256']:
                raise ValueError('member file bytes differ from the pinned SHA: ' + reference['path'])
        test, dev = load_scores(row['test']['path']), load_scores(row['dev']['path'], need_label=True)
        mean, std = float(row['dev_mean']), float(row['dev_std'])
        actual_mean, actual_std = moments(dev['values'])
        if (abs(actual_mean - mean) > 1e-9 or abs(actual_std - std) > 1e-9) and not allow_fixture:
            raise ValueError('declared dev moments differ from the dev file: ' + row['id'])
        if not allow_fixture and len(test['names']) != TEST_ROWS:
            raise ValueError('exactly ' + str(TEST_ROWS) + ' test rows required: ' + row['id'])
        members.append({'id': row['id'], 'weight': float(row.get('weight', 1.0)),
                        'test': test, 'dev': dev, 'dev_mean': mean, 'dev_std': std,
                        'dev_moments_observed': {'mean': actual_mean, 'std': actual_std}})
    if any(member['weight'] <= 0 or not math.isfinite(member['weight']) for member in members):
        raise ValueError('positive finite member weights required')
    reference = members[0]['test']['names']
    if any(member['test']['names'] != reference for member in members[1:]):
        raise ValueError('test image order differs between members')
    if len({tuple(member['dev']['names']) for member in members}) != 1:
        raise ValueError('dev image order differs between members')
    labels = members[0]['dev']['labels']
    if any(not np.array_equal(member['dev']['labels'], labels) for member in members[1:]):
        raise ValueError('dev labels differ between members')
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('two-class dev population required')
    return members, reference, labels


def build(members, names, *, expected_names=None):
    vectors_test, vectors_dev = {}, {}
    for member in members:
        vectors_test[member['id']] = standardise(member['test']['values'], member['dev_mean'], member['dev_std'])
        vectors_dev[member['id']] = standardise(member['dev']['values'], member['dev_mean'], member['dev_std'])
    weights = [member['weight'] for member in members]
    fused_dev = fuse(vectors_dev, weights)
    fused_test = fuse(vectors_test, weights)
    labels = members[0]['dev']['labels']
    parameters = fit_platt(fused_dev, labels)
    probabilities = apply_platt(fused_test, parameters)
    labels_out = (probabilities >= 0.5).astype(int)
    if expected_names is not None and names != list(expected_names):
        raise ValueError('test image names differ from the expected submission order')
    check_submission(names, probabilities, labels_out)
    ranking = {member['id']: spearman(fused_test, member['test']['values']) for member in members}
    return {'test_names': names, 'probabilities': probabilities, 'labels': labels_out,
            'platt': parameters, 'dev_metrics': _metrics(labels, fused_dev),
            'spearman_test_vs_member': ranking,
            'normalization': {member['id']: {'mean': member['dev_mean'], 'std': member['dev_std'],
                                             'weight': member['weight']} for member in members}}


def _metrics(labels, scores):
    """Same tie-aware AUROC and stable AP as the project's dev metric."""
    groups = {}
    for label, score in zip(np.asarray(labels).tolist(), np.asarray(scores, dtype=np.float64).tolist()):
        bucket = groups.setdefault(score, [0, 0])
        bucket[int(label)] += 1
    positives = sum(value[1] for value in groups.values())
    negatives = sum(value[0] for value in groups.values())
    if not positives or not negatives:
        raise ValueError('two-class dev metrics required')
    below = correct = ties = 0
    for score in sorted(groups):
        count, positive = groups[score]
        correct += positive * below
        ties += positive * count
        below += count
    error = 2 * positives * negatives - 2 * correct - ties
    true_positive = false_positive = 0
    last_recall, terms = 0.0, []
    for score in sorted(groups, reverse=True):
        count, positive = groups[score]
        true_positive += positive
        false_positive += count
        recall = true_positive / positives
        terms.append((recall - last_recall) * (true_positive / (true_positive + false_positive)))
        last_recall = recall
    return {'auroc': (2 * positives * negatives - error) / (2 * positives * negatives),
            'ap': math.fsum(terms), 'count': positives + negatives, 'positive': positives}


def check_submission(names, probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if len(names) != probabilities.size or len(names) != labels.size or not len(names):
        raise ValueError('aligned submission columns required')
    if len(set(names)) != len(names):
        raise ValueError('duplicate image names in submission')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('finite probabilities in [0,1] required')
    if not set(np.unique(labels).tolist()) <= {0, 1}:
        raise ValueError('binary predicted labels required')
    if not np.array_equal(labels, (probabilities >= 0.5).astype(int)):
        raise ValueError('predicted labels must equal the 0.5 threshold')
    return True


def write_submission(directory, names, probabilities, labels):
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise ValueError('fresh output directory required')
    directory.mkdir(parents=True)
    path = directory / 'submit.csv'
    with path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Image_name', 'Probability', 'Pred_Label'])
        for name, probability, label in zip(names, probabilities.tolist(), labels.tolist()):
            writer.writerow([name, repr(float(probability)), int(label)])
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'rows': len(names),
            'positive': int(np.asarray(labels).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-names', type=Path)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--allow-fixture', action='store_true')
    args = parser.parse_args()
    if args.threads > 0:
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(args.threads)
        limit = threadpool_limits(args.threads)
    else:
        limit = None
    manifest = json.loads(args.members.read_text())
    members, names, _ = read_members(manifest, allow_fixture=args.allow_fixture)
    expected = None
    if args.expected_names is not None:
        expected = load_scores(args.expected_names)['names']
    result = build(members, names, expected_names=expected)
    written = write_submission(args.output, names, result['probabilities'], result['labels'])
    metadata = {'format': 'f05-fusion-submission-v1', 'status': 'fused_test_submission_written',
                'members': [{'id': member['id'], 'weight': member['weight'],
                             'test': {'path': member['test']['path'], 'sha256': member['test']['sha256']},
                             'dev': {'path': member['dev']['path'], 'sha256': member['dev']['sha256']},
                             'dev_mean': member['dev_mean'], 'dev_std': member['dev_std']}
                            for member in members],
                'manifest': {'path': str(args.members), 'sha256': sha(args.members)},
                'normalization': result['normalization'], 'platt': result['platt'],
                'dev_metrics': result['dev_metrics'],
                'spearman_test_vs_member': result['spearman_test_vs_member'],
                'submission': written, 'rows': written['rows'],
                'official_read': False, 'TTA': False, 'GPU_initialized': False}
    metadata_path = Path(args.output) / 'METADATA.json'
    metadata_path.write_text(json.dumps(metadata, indent=1, sort_keys=True, allow_nan=False) + '\n')
    if limit is not None:
        limit.unregister()
    print(json.dumps({'submission': written, 'platt': result['platt'], 'dev_metrics': result['dev_metrics'],
                      'metadata_sha256': sha(metadata_path)}, sort_keys=True))


if __name__ == '__main__':
    main()
