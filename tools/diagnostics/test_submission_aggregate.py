"""Aggregate-only comparison of submitted test-set probability files.

This tool never reads labels from the official evaluation set and never emits
per-image values; it only reports distribution, ranking and gap summaries as
fixed in ``todo/auroc-999/TEST-SUBMISSION-AGGREGATE-20261105.md``.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

COLUMNS = ['Image_name', 'Probability', 'Pred_Label']
TEST_ROWS = 5855
QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)
SHIFTS = (.10, .20)


def load_submission(path, *, allow_fixture=False):
    """Read one submitted CSV; labels must agree with the 0.5 threshold."""
    names, probabilities, labels = [], [], []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError('submission columns must be exactly ' + ','.join(COLUMNS))
        for row in reader:
            try:
                probability = float(row['Probability'])
                label = int(row['Pred_Label'])
            except (TypeError, ValueError) as error:
                raise ValueError('numeric probability and label required') from error
            names.append(row['Image_name'])
            probabilities.append(probability)
            labels.append(label)
    if not allow_fixture and len(names) != TEST_ROWS:
        raise ValueError('exactly ' + str(TEST_ROWS) + ' test rows required')
    if not names or len(set(names)) != len(names):
        raise ValueError('unique image names required')
    values = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('finite probabilities in [0,1] required')
    if not np.isin(truth, (0, 1)).all():
        raise ValueError('binary predicted labels required')
    if int((values >= .5).sum()) != int(truth.sum()):
        raise ValueError('predicted labels disagree with the 0.5 threshold')
    return {'name': str(path), 'images': names, 'probability': values, 'label': truth}


def require_aligned(submissions):
    """Every submission must list the same images in the same order."""
    if not submissions:
        raise ValueError('at least one submission required')
    names = submissions[0]['images']
    for item in submissions[1:]:
        if item['images'] != names:
            raise ValueError('submitted image order differs')
    return names


def average_ranks(values):
    """Ascending ranks with ties averaged, as in the Spearman convention."""
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


def _pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size != right.size or left.size < 3:
        raise ValueError('paired samples required')
    left_centred, right_centred = left - left.mean(), right - right.mean()
    scale = math.sqrt(float((left_centred ** 2).sum()) * float((right_centred ** 2).sum()))
    if scale == 0:
        raise ValueError('constant sample cannot be correlated')
    return float((left_centred * right_centred).sum() / scale)


def pairwise(submissions):
    """Pearson on probabilities and Spearman on average ranks for every pair."""
    ranks = {item['name']: average_ranks(item['probability']) for item in submissions}
    result = {}
    for index, left in enumerate(submissions):
        for right in submissions[index + 1:]:
            key = left['name'] + '|' + right['name']
            result[key] = {'pearson': _pearson(left['probability'], right['probability']),
                           'spearman': _pearson(ranks[left['name']], ranks[right['name']])}
    return result


def rank_shift(submissions):
    """Share of images whose normalised rank difference exceeds each threshold."""
    ranks = {item['name']: average_ranks(item['probability']) for item in submissions}
    count = submissions[0]['probability'].size
    result = {}
    for index, left in enumerate(submissions):
        for right in submissions[index + 1:]:
            gap = np.abs(ranks[left['name']] - ranks[right['name']]) / float(count)
            result[left['name'] + '|' + right['name']] = {
                'over_' + str(int(threshold * 100)) + 'pct': float((gap > threshold).mean())
                for threshold in SHIFTS}
    return result


def distribution_summary(probability, label):
    probability = np.asarray(probability, dtype=np.float64)
    quantiles = np.quantile(probability, QUANTILES)
    positive = float((probability >= .5).mean())
    label_rate = float(np.asarray(label, dtype=np.float64).mean())
    return {'count': int(probability.size),
            'quantiles': {str(value): float(quantile) for value, quantile in zip(QUANTILES, quantiles)},
            'mean': float(probability.mean()), 'std': float(probability.std(ddof=1)),
            'positive_rate_at_0.5': positive, 'predicted_positive_rate': label_rate,
            'threshold_agreement': float((np.asarray(label) == (probability >= .5)).mean())}


def _ecdf_gap(left, right):
    """Maximum distance between two empirical CDFs (descriptive only)."""
    left, right = np.sort(np.asarray(left)), np.sort(np.asarray(right))
    grid = np.union1d(left, right)
    step_left = np.searchsorted(left, grid, side='right') / left.size
    step_right = np.searchsorted(right, grid, side='right') / right.size
    return float(np.max(np.abs(step_left - step_right)))


def compare_with_reference(probability, reference, label):
    """Describe a submission against an internal reference probability sample."""
    probability, reference = np.asarray(probability, np.float64), np.asarray(reference, np.float64)
    if probability.size < 3 or reference.size < 3:
        raise ValueError('reference and submission need at least three rows')
    if not np.isfinite(probability).all() or not np.isfinite(reference).all():
        raise ValueError('finite probabilities required')
    return {'submission': distribution_summary(probability, label),
            'reference_count': int(reference.size),
            'reference_quantiles': {str(value): float(item)
                                    for value, item in zip(QUANTILES, np.quantile(reference, QUANTILES))},
            'reference_mean': float(reference.mean()), 'reference_std': float(reference.std(ddof=1)),
            'reference_positive_rate_at_0.5': float((reference >= .5).mean()),
            'ecdf_max_difference': _ecdf_gap(probability, reference)}


def load_probabilities(path):
    """Read a reference probability sample from .npy, .npz or a CSV column."""
    path = Path(path)
    if path.suffix == '.npy':
        values = np.load(path, allow_pickle=False)
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            keys = [key for key in archive.files if key.lower() in ('probability', 'probabilities', 'score')]
            if len(keys) != 1:
                raise ValueError('one probability array required in the archive')
            values = archive[keys[0]]
    else:
        values = []
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            field = next((name for name in (reader.fieldnames or [])
                          if name.lower() in ('probability', 'probabilities', 'score')), None)
            if field is None:
                raise ValueError('reference CSV needs a probability column')
            for row in reader:
                values.append(float(row[field]))
        values = np.asarray(values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('finite reference probabilities required')
    return values


def build_report(submissions, *, reference=None, reference_name=None):
    require_aligned(submissions)
    report = {'official_read': False, 'TTA': False, 'rows': int(submissions[0]['probability'].size),
              'submissions': {item['name']: distribution_summary(item['probability'], item['label'])
                              for item in submissions},
              'pairwise': pairwise(submissions), 'rank_shift': rank_shift(submissions)}
    if reference is not None:
        match = [item for item in submissions if item['name'] == reference_name]
        if len(match) != 1:
            raise ValueError('reference submission must be one of the inputs')
        report['against_internal_reference'] = compare_with_reference(
            match[0]['probability'], reference, match[0]['label'])
    return report


def gap_table(entries):
    """Plain internal/official-to-platform AUROC differences, no attribution."""
    result = {}
    for name, item in entries.items():
        source, platform = float(item['source']), float(item['platform'])
        if not 0 <= source <= 1 or not 0 <= platform <= 1:
            raise ValueError('AUROC must be in [0,1]')
        result[name] = {'source': source, 'platform': platform, 'difference': platform - source}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs=2, metavar=('NAME', 'PATH'), action='append', required=True)
    parser.add_argument('--reference', metavar='PATH')
    parser.add_argument('--reference-name')
    parser.add_argument('--gap', nargs=3, metavar=('NAME', 'SOURCE', 'PLATFORM'), action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-fixture', action='store_true')
    args = parser.parse_args()
    submissions = [{**load_submission(path, allow_fixture=args.allow_fixture), 'name': name}
                   for name, path in args.input]
    reference = load_probabilities(args.reference) if args.reference else None
    report = build_report(submissions, reference=reference,
                          reference_name=args.reference_name or ('R120' if reference is not None else None))
    if args.gap:
        report['gap'] = gap_table({name: {'source': source, 'platform': platform}
                                   for name, source, platform in args.gap})
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('fresh output path required')
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'rows': report['rows']}, allow_nan=False))


if __name__ == '__main__':
    main()
