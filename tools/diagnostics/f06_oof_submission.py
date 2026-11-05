"""F06 official-5849 three-fold OOF decision and test submission (CPU only).

Each F06 fold model predicts its own held-out official fold. Held-out logits are
z-normalised with that fold's own mean and population std, concatenated in
official row order into a 5849-row OOF vector, and scored with the project's
tie-aware AUROC/AP. F06 is submitted only when the OOF AUROC strictly beats the
recorded F04 official AUROC; otherwise the existing F04 file is retained. The
test submission standardises each fold model's 5855 test logits with the same
per-fold constants, averages them with equal weight, and applies the
deterministic Platt map fitted on the OOF fused score. No test labels are read,
no TTA is used, and the official labels are used only for this OOF decision.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import fusion_submission as fusion


OFFICIAL_ROWS = 5849
TEST_ROWS = 5855
FOLD_COUNT = 3
F04_OFFICIAL_AUROC = 0.9894371866550443
HEADER = ['Image_name', 'Probability', 'Pred_Label']
NAME_FIELDS = fusion.NAME_FIELDS


def sha(path):
    return fusion.sha(path)


def _fold_key(value):
    if isinstance(value, bool):
        raise ValueError('fold id must be an integer or string')
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or int(value) != value:
            raise ValueError('fold id must be an integer or string')
        return str(int(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError('empty fold id')
        return str(int(text)) if re.fullmatch(r'[+-]?\d+', text) else text
    raise ValueError('fold id must be an integer or string')


def _payload_rows(payload):
    if isinstance(payload, dict):
        rows = payload.get('rows')
        if not isinstance(rows, list):
            raise ValueError('FOLDS object requires a rows list')
        return rows
    if isinstance(payload, list):
        return payload
    raise ValueError('FOLDS must be a row list or an object with rows')


def load_folds(path, *, allow_fixture=False):
    """Read the sealed official fold assignment; every row carries name/label/fold."""
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = _payload_rows(payload)

    names, labels, fold = [], [], []
    seen = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not {'name', 'label', 'fold'} <= set(row):
            raise ValueError('row ' + str(position) + ' requires name, label and fold')
        name = row['name']
        if not isinstance(name, str) or not name.strip():
            raise ValueError('non-empty image name required at row ' + str(position))
        name = name.strip()
        if name in seen:
            raise ValueError('duplicate official image name: ' + name)
        seen.add(name)
        label = row['label']
        if isinstance(label, bool):
            raise ValueError('binary official label required: ' + name)
        try:
            label = int(label)
        except (TypeError, ValueError) as error:
            raise ValueError('binary official label required: ' + name) from error
        if label not in (0, 1):
            raise ValueError('binary official label required: ' + name)
        names.append(name)
        labels.append(label)
        fold.append(_fold_key(row['fold']))

    if not names:
        raise ValueError('at least one official row required')
    if not allow_fixture and len(names) != OFFICIAL_ROWS:
        raise ValueError('exactly ' + str(OFFICIAL_ROWS) + ' official rows required')
    fold_ids = sorted(set(fold))
    if len(fold_ids) != FOLD_COUNT:
        raise ValueError('exactly ' + str(FOLD_COUNT) + ' folds required')
    labels = np.asarray(labels, dtype=np.int64)
    fold_array = np.asarray(fold, dtype=object)
    counts = {}
    for key in fold_ids:
        mask = fold_array == key
        counts[key] = int(mask.sum())
        if set(labels[mask].tolist()) != {0, 1}:
            raise ValueError('fold ' + key + ' requires both official classes')
    return {'path': str(path), 'sha256': sha(path), 'names': names, 'labels': labels,
            'fold': fold, 'fold_ids': fold_ids, 'counts': counts, 'count': len(names)}


def _reference(value, where):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
            or not isinstance(value['path'], str) or not isinstance(value['sha256'], str)
            or not value['path'] or len(value['sha256']) != 64):
        raise ValueError('exact {path, sha256} reference required for ' + where)
    return value


def load_manifest(path, folds):
    """Read the three fold-model inputs and pin every file by its declared SHA."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'f06-oof-members-v1':
        raise ValueError('f06-oof-members-v1 manifest required')
    rows = manifest.get('folds')
    if not isinstance(rows, list) or len(rows) != FOLD_COUNT:
        raise ValueError('exactly ' + str(FOLD_COUNT) + ' fold entries required')
    entries, identifiers = {}, []
    for row in rows:
        if not isinstance(row, dict) or 'fold' not in row:
            raise ValueError('fold entry requires a fold id')
        key = _fold_key(row['fold'])
        identifiers.append(key)
        if key in entries:
            raise ValueError('duplicate fold entry: ' + key)
        oof = _reference(row.get('oof'), 'fold ' + key + ' oof')
        test = row.get('test')
        if test is not None:
            test = _reference(test, 'fold ' + key + ' test')
        column = row.get('score_column')
        if column is not None and not isinstance(column, str):
            raise ValueError('string score_column required')
        entries[key] = {'oof': oof, 'test': test, 'score_column': column}
    if set(identifiers) != set(folds['fold_ids']):
        raise ValueError('manifest folds differ from the FOLDS assignment')
    return {'path': str(path), 'sha256': sha(path), 'entries': entries}


def _header(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return next(csv.reader(stream), [])


def load_predictions(reference, names, *, labels=None, score_column=None, expected_rows=None):
    """Load one fold file, verify its pinned SHA and exact name coverage."""
    if sha(reference['path']) != reference['sha256']:
        raise ValueError('prediction file bytes differ from the pinned SHA: ' + reference['path'])
    loaded = fusion.load_scores(reference['path'], score_column=score_column)
    if expected_rows is not None and len(loaded['names']) != expected_rows:
        raise ValueError('exactly ' + str(expected_rows) + ' rows required in ' + reference['path'])
    if len(loaded['names']) != len(names) or set(loaded['names']) != set(names):
        raise ValueError('fold prediction coverage differs from FOLDS: ' + reference['path'])
    if fusion._field(_header(reference['path']), ('label',)) is not None and labels is not None:
        labelled = fusion.load_scores(reference['path'], need_label=True, score_column=score_column)
        observed = dict(zip(labelled['names'], labelled['labels'].tolist()))
        for name, label in zip(names, np.asarray(labels).tolist()):
            if observed[name] != int(label):
                raise ValueError('official label differs from FOLDS: ' + name)
    position = {name: index for index, name in enumerate(loaded['names'])}
    values = np.asarray([loaded['values'][position[name]] for name in names], dtype=np.float64)
    return {'path': loaded['path'], 'sha256': reference['sha256'], 'values': values,
            'score_field': loaded['score_field'], 'score_scale': loaded['score_scale']}


def build_oof(folds, predictions):
    """Concatenate per-fold held-out z scores and score overall and per fold."""
    labels = folds['labels']
    fold_array = np.asarray(folds['fold'], dtype=object)
    z = np.empty(folds['count'], dtype=np.float64)
    moments_by_fold, per_fold = {}, {}
    for key in folds['fold_ids']:
        mask = fold_array == key
        values = predictions[key]['values']
        mean, std = fusion.moments(values)
        z[mask] = fusion.standardise(values, mean, std)
        moments_by_fold[key] = {'mean': mean, 'std': std, 'count': int(mask.sum())}
        per_fold[key] = {'z': moments_by_fold[key],
                         'metrics': fusion._metrics(labels[mask], z[mask])}
    return {'z': z, 'moments': moments_by_fold, 'per_fold': per_fold,
            'metrics': fusion._metrics(labels, z)}


def decide(auroc):
    """Strict comparison: F04 is retained unless the OOF AUROC is above it."""
    return 'F06' if float(auroc) > F04_OFFICIAL_AUROC else 'F04'


def load_reference(path, *, allow_fixture=False):
    """Read the frozen submission order (names only); never uses the score column."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if header != HEADER:
            raise ValueError('reference header must be ' + ','.join(HEADER))
        names = []
        for position, row in enumerate(reader):
            if not row or not row[0].strip():
                raise ValueError('empty reference name at row ' + str(position + 1))
            names.append(row[0].strip())
    if not allow_fixture and len(names) != TEST_ROWS:
        raise ValueError('exactly ' + str(TEST_ROWS) + ' reference rows required')
    if not names or len(set(names)) != len(names):
        raise ValueError('unique reference image names required')
    return {'path': str(path), 'sha256': sha(path), 'names': names, 'rows': len(names)}


def build_submission(oof, folds, test_predictions, names):
    """Equal-weight z fusion of the three fold models, Platt-mapped on the OOF score."""
    vectors = {}
    for key in folds['fold_ids']:
        statistics = oof['moments'][key]
        vectors[key] = fusion.standardise(test_predictions[key]['values'],
                                          statistics['mean'], statistics['std'])
    fused_test = fusion.fuse(vectors, [1.0] * len(vectors))
    parameters = fusion.fit_platt(oof['z'], folds['labels'])
    probabilities = fusion.apply_platt(fused_test, parameters)
    labels_out = (probabilities >= 0.5).astype(np.int64)
    fusion.check_submission(names, probabilities, labels_out)
    return {'names': list(names), 'probabilities': probabilities, 'labels': labels_out,
            'platt': parameters, 'fused_test': fused_test}


def _write_csv(path, names, probabilities, labels):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\r\n')
        writer.writerow(HEADER)
        for name, probability, label in zip(names, probabilities.tolist(), labels.tolist()):
            writer.writerow([name, repr(float(probability)), int(label)])


def write_output(directory, metadata, submission=None):
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise ValueError('fresh output directory required')
    directory.mkdir(parents=True)
    written = None
    if submission is not None:
        path = directory / 'submit.csv'
        _write_csv(path, submission['names'], submission['probabilities'], submission['labels'])
        written = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                   'rows': len(submission['names']),
                   'positive': int(np.asarray(submission['labels']).sum())}
    metadata['submission'] = written
    (directory / 'METADATA.json').write_text(
        json.dumps(metadata, indent=1, sort_keys=True, allow_nan=False) + '\n')
    return written


def execute(folds_path, manifest_path, output, reference_path=None, *, oof_only=False,
            allow_fixture=False):
    folds = load_folds(folds_path, allow_fixture=allow_fixture)
    manifest = load_manifest(manifest_path, folds)
    fold_array = np.asarray(folds['fold'], dtype=object)
    fold_names = {key: [name for name, value in zip(folds['names'], folds['fold']) if value == key]
                  for key in folds['fold_ids']}
    fold_labels = {key: folds['labels'][fold_array == key] for key in folds['fold_ids']}
    predictions = {}
    for key in folds['fold_ids']:
        predictions[key] = load_predictions(manifest['entries'][key]['oof'], fold_names[key],
                                            labels=fold_labels[key],
                                            score_column=manifest['entries'][key]['score_column'])
    oof = build_oof(folds, predictions)
    decision = decide(oof['metrics']['auroc'])

    reference, test_predictions, submission = None, {}, None
    if not oof_only:
        if reference_path is None:
            raise ValueError('--reference is required unless --oof-only is used')
        reference = load_reference(reference_path, allow_fixture=allow_fixture)
        if decision == 'F06':
            for key in folds['fold_ids']:
                entry = manifest['entries'][key]
                if entry['test'] is None:
                    raise ValueError('test prediction reference required for fold ' + key)
                test_predictions[key] = load_predictions(
                    entry['test'], reference['names'],
                    score_column=entry['score_column'],
                    expected_rows=None if allow_fixture else TEST_ROWS)
            submission = build_submission(oof, folds, test_predictions, reference['names'])

    status = ('oof_only' if oof_only else
              'f06_submission_written' if decision == 'F06' else 'f04_retained_no_submission')
    metadata = {
        'format': 'f06-oof-submission-v1', 'status': status, 'decision': decision,
        'decision_rule': {'metric': 'oof_auroc', 'comparison': '>',
                          'f04_official_auroc': F04_OFFICIAL_AUROC,
                          'observed_oof_auroc': oof['metrics']['auroc'],
                          'f06_selected': decision == 'F06'},
        'oof': {'count': folds['count'], 'positive': int(folds['labels'].sum()),
                'metrics': oof['metrics'], 'per_fold': oof['per_fold']},
        'folds': {'path': folds['path'], 'sha256': folds['sha256'], 'counts': folds['counts']},
        'manifest': {'path': manifest['path'], 'sha256': manifest['sha256']},
        'models': [{'fold': key,
                    'oof': {'path': predictions[key]['path'],
                            'sha256': predictions[key]['sha256'],
                            'score_field': predictions[key]['score_field'],
                            'score_scale': predictions[key]['score_scale']},
                    'z': oof['moments'][key],
                    'test': ({'path': test_predictions[key]['path'],
                              'sha256': test_predictions[key]['sha256'],
                              'score_field': test_predictions[key]['score_field'],
                              'score_scale': test_predictions[key]['score_scale']}
                             if key in test_predictions else None)}
                   for key in folds['fold_ids']],
        'reference_order': ({'path': reference['path'], 'sha256': reference['sha256'],
                             'rows': reference['rows']} if reference is not None else None),
        'platt': (submission['platt'] if submission is not None else None),
        'platt_fit': 'oof_fused_z',
        'submission': None,
        'official_read': True,
        'official_read_scope': 'oof_decision_only',
        'test_label_values_read': False,
        'TTA': False,
        'GPU_initialized': False,
    }
    written = write_output(Path(output), metadata, submission)
    return metadata, written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oof-only', action='store_true')
    parser.add_argument('--allow-fixture', action='store_true')
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()
    if args.threads > 0:
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(args.threads)
        limit = threadpool_limits(args.threads)
    else:
        limit = None
    metadata, written = execute(args.folds, args.manifest, args.output, args.reference,
                                oof_only=args.oof_only, allow_fixture=args.allow_fixture)
    if limit is not None:
        limit.unregister()
    print(json.dumps({'status': metadata['status'], 'decision': metadata['decision'],
                      'oof_metrics': metadata['oof']['metrics'],
                      'per_fold': {key: value['metrics']
                                   for key, value in metadata['oof']['per_fold'].items()},
                      'submission': written,
                      'metadata_sha256': sha(Path(args.output) / 'METADATA.json')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
