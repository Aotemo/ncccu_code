"""Audit existing selected predictions without training, image IO or new forwards.

All computation belongs on the existing guarded Pod CPU. Inventories and sample
identities stay in exp-v3; the report contains aggregate evidence, never a new
checkpoint selection or a training admission decision.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
sys.path.insert(0, str(Path(__file__).parent))
from persistent_errors import pair_contributions
from retrain.n05_metrics import metrics

LIMITS = [
    'Intervals condition on the supplied trained and selected checkpoints; they do not include seed variability.',
    'Resampling does not remove adaptive selection bias from repeated use of the same dev set.',
    'Image-level intervals are not patient-level intervals; supplied group IDs are not independently verified patient IDs.',
    'Pairwise errors share samples; positive-negative pairs are not independent bootstrap observations.',
    'Intervals are descriptive, unadjusted for multiple comparisons, and do not authorize training or change selected checkpoints.',
    'Fit/dev gaps and supplied structure annotations alone do not identify a clinical cause, label error or achievable AUROC ceiling.',
]


def _ranking(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (y.ndim != 1 or s.shape != y.shape or not y.size
            or set(y.tolist()) != {0, 1} or not np.isfinite(s).all()):
        raise ValueError('finite aligned binary scores with both classes required')
    order = np.argsort(s, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(s[order]) != 0) + 1]
    return order, starts, y[order].astype(np.float64)


def _auc_from_weights(ranking, weights):
    order, starts, y = ranking
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w[None, :]
    if (w.ndim != 2 or w.shape[1] != len(order) or not np.isfinite(w).all()
            or (w < 0).any()):
        raise ValueError('finite nonnegative aligned sample weights required')
    ordered = w[:, order]
    p = np.add.reduceat(ordered * y, starts, axis=1)
    n = np.add.reduceat(ordered * (1 - y), starts, axis=1)
    denominator = p.sum(axis=1) * n.sum(axis=1)
    if (denominator <= 0).any():
        raise ValueError('every resample needs both classes')
    below = np.cumsum(n, axis=1) - n
    return (p * (below + .5 * n)).sum(axis=1) / denominator


def weighted_auc(labels, scores, weights=None):
    """One AUC per row of sample counts; exact ties receive half credit."""
    ranking = _ranking(labels, scores)
    return _auc_from_weights(ranking, np.ones(len(labels)) if weights is None else weights)


def paired_bootstrap(labels, reference, candidate, *, replicates=2000, seed=20261103,
                     confidence=.95, groups=None):
    if (type(replicates) is not int or not 2 <= replicates <= 20000
            or type(seed) is not int or seed < 0 or not 0 < confidence < 1):
        raise ValueError('bounded replicate count, nonnegative integer seed and confidence required')
    y = np.asarray(labels)
    a, b = _ranking(labels, reference), _ranking(labels, candidate)
    rng = np.random.Generator(np.random.PCG64(seed))
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    inverse = None
    if groups is not None:
        if len(groups) != len(y) or any(not isinstance(g, str) or not g for g in groups):
            raise ValueError('one nonempty group identity per sample required')
        unique, inverse = np.unique(groups, return_inverse=True)
        if len(unique) < 2:
            raise ValueError('at least two supplied groups required')
    deltas, accepted, drawn = [], 0, 0
    while accepted < replicates and drawn < 10 * replicates:
        count = min(64, replicates - accepted, 10 * replicates - drawn)
        if inverse is None:
            weights = np.zeros((count, len(y)), dtype=np.float64)
            weights[:, positives] = rng.multinomial(len(positives), np.full(len(positives), 1 / len(positives)), size=count)
            weights[:, negatives] = rng.multinomial(len(negatives), np.full(len(negatives), 1 / len(negatives)), size=count)
        else:
            clusters = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=count)
            weights = clusters[:, inverse]
        drawn += count
        valid = (weights[:, positives].sum(1) > 0) & (weights[:, negatives].sum(1) > 0)
        weights = weights[valid]
        if len(weights):
            deltas.append(_auc_from_weights(b, weights) - _auc_from_weights(a, weights))
            accepted += len(weights)
    if accepted != replicates:
        raise ValueError('not enough valid paired group resamples; no image-level fallback')
    samples = np.concatenate(deltas)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha], method='linear')
    delta = float(weighted_auc(labels, candidate)[0] - weighted_auc(labels, reference)[0])
    return {'delta_auroc': delta, 'interval': [float(low), float(high)],
            'confidence': confidence, 'replicates': replicates, 'draws': drawn,
            'method': 'paired_percentile', 'quantile_method': 'linear',
            'invalid_single_class_draws': drawn - accepted, 'seed': seed, 'generator': 'PCG64',
            'unit': 'stratified_image' if inverse is None else 'supplied_group',
            'group_count': None if inverse is None else len(unique),
            'fraction_delta_gt_zero': float(np.mean(samples > 0)),
            'direction': 'conditional_positive' if low > 0 else 'conditional_negative' if high < 0 else 'uncertain',
            'selection_bias_corrected': False}


def _pinned_bytes(spec):
    path = Path(spec['path'])
    wanted = spec.get('sha256')
    if (not path.is_absolute() or path.is_symlink() or not isinstance(wanted, str)
            or re.fullmatch('[0-9a-f]{64}', wanted) is None):
        raise ValueError('absolute nonsymlink file and explicit SHA256 required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != wanted:
        raise ValueError('pinned input SHA256 differs')
    return raw


def _pinned_json(spec):
    return json.loads(_pinned_bytes(spec))


def _reference(spec):
    manifest = _pinned_json(spec)
    rows, dev, fit = manifest['rows'], manifest['dev_indices'], manifest['fit_indices']
    if (not dev or not fit or any(type(i) is not int for i in dev + fit)
            or len(set(dev + fit)) != len(dev + fit) or set(dev + fit) != set(range(len(rows)))
            or any(type(r['label']) is not int or r['label'] not in (0, 1)
                   or len(Path(r['relative_path']).parts) != 2
                   or Path(r['relative_path']).parts[0] != 'train'
                   or Path(r['relative_path']).name.casefold() != r['name'].casefold() for r in rows)
            or len({r['name'].casefold() for r in rows}) != len(rows)):
        raise ValueError('sealed disjoint fit/internal-dev train-source manifest required')
    return {'rows': rows, 'dev': [rows[i] for i in dev], 'fit': fit,
            'manifest_sha256': spec['sha256']}


def _prediction_path(path, variant, split):
    path = Path(path)
    resolved = path.resolve(strict=True)
    pattern = r'dev-cycle-\d{4}\.csv' if split == 'dev' else r'fit-(selected|cycle-\d{4})\.csv'
    if (variant not in ('ema', 'online') or path.is_symlink()
            or resolved.parent.name != variant or resolved.parent.parent.name != 'predictions'
            or re.fullmatch(pattern, resolved.name) is None):
        raise ValueError('only explicit internal dev or fit prediction paths allowed')


def _scores(spec, rows, variant, split):
    _prediction_path(spec['path'], variant, split)
    records = list(csv.DictReader(io.StringIO(_pinned_bytes(spec).decode('utf-8-sig'))))
    if len(records) != len(rows):
        raise ValueError('prediction count differs from declared manifest population')
    scores = []
    for item, row in zip(records, rows):
        if item['name'] != row['name'] or item['label'] != str(row['label']):
            raise ValueError('prediction name/order/label differs from manifest')
        score = float(item['logit'])
        if not math.isfinite(score):
            raise ValueError('nonfinite prediction')
        scores.append(score)
    return scores


def _precision(spec, selected):
    proof = _pinned_json(spec)
    if (proof.get('variant') != selected['variant'] or proof.get('view') != 'identity'
            or proof.get('actual_tensor_sha256') != selected['tensor_sha256']
            or proof.get('state_identity', {}).get('successful_updates') != selected['successful_updates']):
        raise ValueError('prediction precision/checkpoint identity differs from selected state')


def _selected(spec, reference):
    freeze = _pinned_json(spec['freeze'])
    selected = freeze['selected']
    completion = _pinned_json(spec['completion'])
    total = completion.get('successful_updates')
    round_root = Path(spec['freeze']['path']).parent
    if (Path(spec['completion']['path']).name != 'TRAIN_RESULT.json'
            or Path(spec['completion']['path']).parent not in (round_root, round_root / 'evidence')
            or completion.get('status') != 'completed' or type(total) is not int
            or type(selected.get('successful_updates')) is not int
            or not 0 < selected['successful_updates'] <= total):
        raise ValueError('completed same-round TRAIN_RESULT with valid selected update required')
    if (freeze.get('cpu_fixture_only') is not False or freeze.get('precision') != 'fp32'
            or freeze.get('view') != 'identity'
            or freeze.get('experiment91', {}).get('manifest_sha256') != reference['manifest_sha256']
            or re.fullmatch('[0-9a-f]{64}', selected['tensor_sha256']) is None):
        raise ValueError('complete selected internal-dev freeze/manifest identity required')
    prediction = {'path': selected['prediction_path'], 'sha256': selected['prediction_sha256']}
    precision = {'path': selected['precision_path'], 'sha256': selected['precision_sha256']}
    if precision['path'] != prediction['path'] + '.precision.json':
        raise ValueError('selected precision path differs')
    scores = _scores(prediction, reference['dev'], selected['variant'], 'dev')
    _precision(precision, selected)
    actual = metrics([{'label': r['label'], 'logit': s} for r, s in zip(reference['dev'], scores)])
    if any(abs(actual[key] - selected[key]) > 1e-12 for key in actual):
        raise ValueError('recomputed metrics differ from original selected metrics')
    fit = {'status': 'unknown', 'gap': None, 'reason': 'no_same_selected_fit_predictions'}
    fit_spec = spec.get('fit_prediction')
    if fit_spec is not None:
        indices = fit_spec['manifest_indices']
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or i not in reference['fit'] for i in indices)):
            raise ValueError('fit predictions must use only declared fit indices')
        fit_scores = _scores(fit_spec, [reference['rows'][i] for i in indices], selected['variant'], 'fit')
        _precision(fit_spec['precision'], selected)
        fit_metrics = metrics([{'label': reference['rows'][i]['label'], 'logit': s}
                               for i, s in zip(indices, fit_scores)])
        fit = {'status': 'available', 'count': len(indices), 'complete_fit': set(indices) == set(reference['fit']),
               'metrics': fit_metrics, 'gap': fit_metrics['auroc'] - actual['auroc'],
               'prediction': {'path': fit_spec['path'], 'sha256': fit_spec['sha256']},
               'diagnosis': 'gap_is_descriptive_not_a_causal_diagnosis'}
    summary = {**actual, 'variant': selected['variant'], 'cycle': selected['cycle'],
               'successful_updates': selected['successful_updates'], 'tensor_sha256': selected['tensor_sha256'],
               'freeze': spec['freeze'], 'prediction': prediction, 'precision': precision,
               'completion': {**spec['completion'], 'status': 'completed', 'successful_updates': total}}
    return scores, summary, fit


def _groups(spec, rows):
    if spec is None:
        return None
    values = list(csv.DictReader(io.StringIO(_pinned_bytes(spec).decode('utf-8-sig'))))
    if len(values) != len(rows) or any(v['name'] != r['name'] or not v['group_id'].strip()
                                      for v, r in zip(values, rows)):
        raise ValueError('explicit group table must match every dev row in order')
    return [v['group_id'] for v in values]


def _structure_review(spec, reference):
    if spec is None:
        return {'status': 'unknown', 'reviewed_count': None, 'reason': 'no_source_bound_human_structure_review'}
    values = list(csv.DictReader(io.StringIO(_pinned_bytes(spec).decode('utf-8-sig'))))
    source = {r['name']: r for r in reference['rows']}
    if not values or len({v['name'] for v in values}) != len(values):
        raise ValueError('nonempty unique structure-review rows required')
    for row in values:
        if (row['name'] not in source or row['original_sha256'] != source[row['name']]['sha256']
                or re.fullmatch('[0-9a-f]{64}', row['prepared_sha256']) is None
                or row['original_disc_visible'] not in ('yes', 'no', 'unknown')
                or row['prepared_disc_visible'] not in ('yes', 'no', 'unknown')
                or row['label_review'] not in ('agree', 'disagree', 'uncertain') or not row['reviewer'].strip()):
            raise ValueError('structure-review source identity or observation differs')
    return {'status': 'available', 'reviewed_count': len(values), 'source': spec,
            'potential_visibility_loss': sum(v['original_disc_visible'] == 'yes'
                                             and v['prepared_disc_visible'] == 'no' for v in values),
            'label_disagreement_count': sum(v['label_review'] == 'disagree' for v in values),
            'uncertain_label_count': sum(v['label_review'] == 'uncertain' for v in values),
            'images_independently_checked': False, 'labels_changed': False}


def _error_changes(labels, reference, candidate, fixed):
    positives, negatives = np.flatnonzero(np.asarray(labels) == 1), np.flatnonzero(np.asarray(labels) == 0)
    def errors(scores):
        x = np.asarray(scores)
        p, n = x[positives, None], x[negatives][None, :]
        return (p < n).astype(np.float64) + .5 * (p == n)
    a, b = errors(reference), errors(candidate)
    masks = {'all': np.ones(len(positives), dtype=bool)}
    if fixed is not None:
        masks['fixed_positive'] = np.isin(positives, fixed)
        masks['remaining_positive'] = ~masks['fixed_positive']
    output = {}
    for name, mask in masks.items():
        left, right = a[mask], b[mask]
        output[name] = {'reference': float(left.sum()), 'candidate': float(right.sum()),
                        'repaired': float(np.maximum(left - right, 0).sum()),
                        'introduced': float(np.maximum(right - left, 0).sum()),
                        'shared': float(np.minimum(left, right).sum()), 'net_added': float((right - left).sum())}
    return output


def analyze(inventory):
    if len(inventory['models']) < 2 or not inventory['comparisons']:
        raise ValueError('at least two selected models and one predeclared comparison required')
    reference = _reference(inventory['manifest'])
    labels = [r['label'] for r in reference['dev']]
    fixed = inventory.get('fixed_positive_indices')
    if fixed is not None and (not fixed or len(set(fixed)) != len(fixed)
                              or any(type(i) is not int or not 0 <= i < len(labels) or labels[i] != 1 for i in fixed)):
        raise ValueError('fixed positive group must be predeclared unique positive dev indices')
    groups = _groups(inventory.get('bootstrap_groups'), reference['dev'])
    vectors, summaries, fit = {}, {}, {}
    for spec in inventory['models']:
        name = spec['id']
        if not isinstance(name, str) or not name or name in vectors:
            raise ValueError('unique nonempty model identities required')
        vectors[name], summaries[name], fit[name] = _selected(spec, reference)
        mass = pair_contributions(labels, vectors[name])
        if any(sum(m for m, y in zip(mass, labels) if y == label) != summaries[name]['error_twice'] for label in (0, 1)):
            raise ValueError('error contributions do not conserve exact pair mass')
    comparisons = []
    for candidate, baseline in inventory['comparisons']:
        if candidate == baseline or candidate not in vectors or baseline not in vectors:
            raise ValueError('predeclared comparisons require two loaded model identities')
        bootstrap = paired_bootstrap(labels, vectors[baseline], vectors[candidate], groups=groups,
                                      **inventory.get('bootstrap', {}))
        changes = _error_changes(labels, vectors[baseline], vectors[candidate], fixed)
        if any(changes['all'][key] != summaries[name]['error_twice'] / 2
               for key, name in (('reference', baseline), ('candidate', candidate))):
            raise ValueError('comparison pair mass differs from selected metrics')
        comparisons.append({'candidate': candidate, 'reference': baseline, **bootstrap, 'error_mass': changes})
    return {'schema_version': 1, 'status': 'evidence_report',
            'manifest': inventory['manifest'], 'count': len(labels), 'positives': labels.count(1),
            'negatives': labels.count(0), 'models': summaries, 'comparisons': comparisons,
            'fixed_positive_count': None if fixed is None else len(fixed),
            'fixed_positive_group_source': inventory.get('fixed_positive_group_source'),
            'bootstrap_groups': inventory.get('bootstrap_groups'),
            'evidence': {'fit_dev': fit, 'structure': _structure_review(inventory.get('structure_review'), reference)},
            'causal_diagnosis': 'not_identified', 'limits': LIMITS,
            'checkpoint_selection_changed': False, 'official_read': False,
            'model_loaded': False, 'new_forward': False, 'labels_changed': False,
            'numpy_version': np.__version__,
            'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__), Path(__file__).with_name('persistent_errors.py'),
                                           ROOT / 'runtime/bundle/retrain/n05_metrics.py')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if (not os.environ.get('NCCCU_EXECUTOR_ID') or not os.environ.get('NCCCU_CPU_ALLOCATION')
            or any(os.environ.get(k) != '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))):
        raise ValueError('run only through the allocated hidden-GPU Pod CPU guard')
    if (not args.output.is_absolute() or 'exp-v3' not in args.output.parts
            or args.output.exists() or not args.output.parent.is_dir()):
        raise ValueError('new private output file under an existing exp-v3 directory required')
    spec = {'path': str(args.inventory), 'sha256': args.inventory_sha256}
    report = analyze(_pinned_json(spec))
    report['inventory'] = spec
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output),
                      'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'count': report['count'], 'comparisons': report['comparisons'],
                      'fit_status': {k: v['status'] for k, v in report['evidence']['fit_dev'].items()},
                      'structure_status': report['evidence']['structure']['status']}, indent=2))


if __name__ == '__main__':
    main()
