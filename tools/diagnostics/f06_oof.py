"""F06 honest official OOF evaluation, test inference and three-fold fused submission (CPU/GPU)."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parents[2] / 'runtime' / 'bundle'
sys.path.insert(0, str(BUNDLE))
F04_AUROC = 0.9894371866550443
TEST_ROWS = 5855


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as stream:
        stream.write(text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_manifest(args):
    from retrain.data import load_source
    source = load_source(args.data_root, args.train_identity, True, True, 51396)
    rows = source['valid_rows']
    if len(rows) != 5849 or sum(int(r['label']) for r in rows) != 191:
        raise SystemExit('official manifest population differs')
    value = {'format': 'official-valid-monitoring-v1', 'training_eligible': False,
             'source_identity': {'valid_csv_sha256': source['identity']['valid_csv_sha256'],
                                 'valid_rows_sha256': source['identity']['valid_rows_sha256'],
                                 'valid_count': 5849},
             'rows': rows}
    text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n'
    print(json.dumps({'status': 'f06_official_manifest_written', 'path': args.output,
                      'sha256': write_new(args.output, text), 'rows': len(rows),
                      'positive': sum(int(r['label']) for r in rows)}, indent=1))


def cmd_evaluate(args):
    from retrain import inference
    metadata = json.loads((Path(args.export) / 'model.json').read_text())
    variant = metadata['checkpoint']['variant']
    report = inference.run_evaluation(args.export, args.data_root, args.csv_path,
                                      device=args.device, precision='fp32', expected_count=5849,
                                      variant=variant, eval_manifest=args.manifest,
                                      eval_manifest_sha256=sha(args.manifest))
    rows = list(csv.DictReader(Path(args.csv_path).open(newline='')))
    if len(rows) != 5849 or len({r['name'] for r in rows}) != 5849:
        raise SystemExit('evaluation output coverage differs')
    receipt = {'status': 'f06_fold_official_scored', 'fold': args.fold, 'export': args.export,
               'variant': variant, 'csv': args.csv_path, 'csv_sha256': sha(args.csv_path),
               'metrics': report['metrics'], 'input_csv_sha256': report['input_csv_sha256'],
               'eval_manifest_sha256': report['eval_manifest_sha256'],
               'GPU_initialized': True, 'test_labels_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(args.receipt, json.dumps(receipt, indent=1, sort_keys=True) + '\n')}, indent=1))


def cmd_test(args):
    from retrain import inference
    metadata = json.loads((Path(args.export) / 'model.json').read_text())
    variant = metadata['checkpoint']['variant']
    captured = []
    original = inference._forward

    def forward(model, images, device, precision, age_features=None):
        import torch
        with torch.inference_mode(), inference._autocast(device, precision):
            images = images.to(device)
            logits = (model(images, age_features.to(device, dtype=torch.float32))
                      if age_features is not None else model(images)).reshape(-1).float()
        if logits.numel() != images.shape[0] or not torch.isfinite(logits).all():
            raise FloatingPointError('invalid test logits')
        captured.extend(logits.cpu().tolist())
        return logits.sigmoid().cpu().tolist()

    inference._forward = forward
    try:
        report = inference.run_prediction(args.export, args.data_root, args.csv_path,
                                          device=args.device, variant=variant, expected_count=TEST_ROWS,
                                          threshold=.5, batch_size=32, precision='fp32')
    finally:
        inference._forward = original
    if len(captured) != TEST_ROWS:
        raise SystemExit('test logit count differs')
    names = list(csv.DictReader(Path(args.csv_path).open(newline='')))
    if len(names) != TEST_ROWS:
        raise SystemExit('test probability rows differ')
    logits = args.logits_path
    Path(logits).parent.mkdir(parents=True, exist_ok=True)
    with Path(logits).open('x', newline='', encoding='utf-8') as stream:
        stream.write('Image_name,Logit\n')
        for row, value in zip(names, captured):
            if not math.isfinite(value):
                raise SystemExit('nonfinite test logit')
            stream.write(f"{row['Image_name']},{value!r}\n")
    receipt = {'status': 'f06_fold_test_scored', 'fold': args.fold, 'export': args.export,
               'variant': variant, 'probabilities': args.csv_path, 'probabilities_sha256': sha(args.csv_path),
               'logits': str(logits), 'logits_sha256': sha(logits), 'rows': len(captured),
               'TTA': False, 'test_labels_read': False}
    print(json.dumps(receipt | {'receipt_sha256': write_new(args.receipt, json.dumps(receipt, indent=1, sort_keys=True) + '\n')}, indent=1))


def cmd_filter(args):
    """Keep only the held-out fold rows of a full official prediction CSV."""
    folds = json.loads(Path(args.folds).read_text())
    held = [row for row in folds['rows'] if row['fold'] == args.fold]
    rows = list(csv.DictReader(Path(args.full).open(newline='')))
    if len(rows) != 5849:
        raise SystemExit('full official prediction rows differ')
    by_name = {row['name'].casefold(): row for row in rows}
    if len(by_name) != 5849:
        raise SystemExit('duplicate names in full prediction')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['name', 'label', 'logit'])
        for row in held:
            match = by_name.get(row['name'].casefold())
            if match is None or int(match['label']) != int(row['label']):
                raise SystemExit('held-out coverage/label differs: ' + row['name'])
            writer.writerow([row['name'], int(row['label']), match['logit']])
    receipt = {'status': 'f06_fold_heldout_extracted', 'fold': args.fold,
               'full': str(args.full), 'full_sha256': sha(args.full),
               'heldout': str(path), 'heldout_sha256': sha(path),
               'rows': len(held), 'positive': sum(int(row['label']) for row in held)}
    print(json.dumps(receipt | {'receipt_sha256': write_new(args.receipt, json.dumps(receipt, indent=1, sort_keys=True) + '\n')}, indent=1))


def _read_scores(path, *, need_label):
    names, values, labels = [], [], []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = {name.lower(): name for name in reader.fieldnames or []}
        name_field = fields.get('image_name') or fields.get('name')
        logit_field = fields.get('logit')
        label_field = fields.get('label')
        if name_field is None or logit_field is None or (need_label and label_field is None):
            raise SystemExit('name/logit/label columns required: ' + str(path))
        for row in reader:
            names.append(row[name_field])
            values.append(float(row[logit_field]))
            if need_label:
                labels.append(int(row[label_field]))
    if need_label and set(labels) != {0, 1}:
        raise SystemExit('two-class labels required')
    return names, values, labels


def _metrics(labels, scores):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {'auroc': float(roc_auc_score(labels, scores)),
            'ap': float(average_precision_score(labels, scores))}


def cmd_fuse(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    folds = json.loads(Path(args.folds).read_text())
    fold_of = {row['name'].casefold(): row['fold'] for row in folds['rows']}
    index_of = {row['name'].casefold(): i for i, row in enumerate(folds['rows'])}
    oof = [None] * 5849
    moments = {}
    names_order = None
    for fold in range(3):
        names, values, labels = _read_scores(args.oof[fold], need_label=True)
        if len(names) != 5849:
            raise SystemExit('OOF vector row count differs')
        values = np.asarray(values, dtype=np.float64)
        held = np.asarray([fold_of[name.casefold()] == fold for name in names])
        if int(held.sum()) == 0:
            raise SystemExit('empty held-out fold')
        subset = values[held]
        mean, std = float(subset.mean()), float(subset.std(ddof=0))
        if not std > 0:
            raise SystemExit('constant held-out fold scores')
        moments[fold] = {'mean': mean, 'std': std, 'rows': int(held.sum())}
        for name, value in zip(names, values):
            if name.casefold() in fold_of and fold_of[name.casefold()] == fold:
                index = index_of[name.casefold()]
                if oof[index] is not None:
                    raise SystemExit('duplicate OOF row')
                oof[index] = (value - mean) / std
        if names_order is None:
            names_order = names
    if any(value is None for value in oof):
        raise SystemExit('incomplete official OOF coverage')
    labels = [row['label'] for row in folds['rows']]
    fused_oof = np.asarray(oof, dtype=np.float64)
    metrics = _metrics(labels, fused_oof)
    result = {'status': 'f06_oof_measured', 'count': len(labels), 'positive': sum(labels),
              'oof_auroc': metrics['auroc'], 'oof_ap': metrics['ap'], 'f04_official_auroc': F04_AUROC,
              'beats_f04': metrics['auroc'] > F04_AUROC, 'fold_moments': {str(k): v for k, v in moments.items()},
              'oof_files': [{'path': str(p), 'sha256': sha(p)} for p in args.oof]}
    if not result['beats_f04']:
        result['submission'] = 'F04-existing-file'
        print(json.dumps(result | {'receipt_sha256': write_new(args.receipt, json.dumps(result, indent=1, sort_keys=True) + '\n')}, indent=1))
        return
    test_vectors = []
    test_names = None
    for fold in range(3):
        names, values, _ = _read_scores(args.test[fold], need_label=False)
        if len(names) != TEST_ROWS:
            raise SystemExit('test vector row count differs')
        if test_names is None:
            test_names = names
        elif names != test_names:
            raise SystemExit('test image order differs between folds')
        mean, std = moments[fold]['mean'], moments[fold]['std']
        test_vectors.append((np.asarray(values, dtype=np.float64) - mean) / std)
    fused_test = np.mean(np.stack(test_vectors), axis=0)
    model = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, tol=1e-10, random_state=20261105)
    model.fit(fused_oof.reshape(-1, 1), np.asarray(labels))
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    if not (math.isfinite(slope) and math.isfinite(intercept) and slope > 0):
        raise SystemExit('invalid Platt parameters')
    probabilities = 1.0 / (1.0 + np.exp(-(slope * fused_test + intercept)))
    if not np.isfinite(probabilities).all():
        raise SystemExit('nonfinite submission probabilities')
    labels_out = (probabilities >= 0.5).astype(int)
    expected = [row['Image_name'] for row in csv.DictReader(Path(args.reference).open(newline=''))]
    if test_names != expected:
        raise SystemExit('test order differs from the reference submission')
    out = Path(args.submit)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\r\n')
        writer.writerow(['Image_name', 'Probability', 'Pred_Label'])
        for name, probability, label in zip(test_names, probabilities, labels_out):
            writer.writerow([name, f'{float(probability):.10f}', int(label)])
    rows = list(csv.DictReader(out.open(newline='')))
    if len(rows) != TEST_ROWS or [r['Image_name'] for r in rows] != expected:
        raise SystemExit('submitted row count/order differs')
    if any(not 0 <= float(r['Probability']) <= 1 for r in rows):
        raise SystemExit('submission probability out of range')
    metadata = {'status': 'f06_three_fold_official_finetuned_submission',
                'principal': 'F06 official-5849 three-fold fine-tuning',
                'members': [{'fold': fold, 'export': args.exports[fold], 'oof': args.oof[fold],
                             'test': args.test[fold]} for fold in range(3)],
                'folds': {'path': str(args.folds), 'sha256': sha(args.folds)},
                'fusion': {'scale': 'logit', 'weighting': 'equal', 'normalization': 'per_fold_heldout_moments',
                           'moments': moments},
                'platt': {'slope': slope, 'intercept': intercept, 'fit': 'official_oof_fused_score'},
                'oof': {'auroc': metrics['auroc'], 'ap': metrics['ap'], 'count': len(labels),
                        'positive': sum(labels)},
                'f04_official_auroc': F04_AUROC, 'TTA': False, 'official_used_for_selection': True,
                'test_labels_read': False,
                'submission': {'path': str(out), 'sha256': sha(out), 'rows': len(rows),
                               'predicted_positive': int(labels_out.sum())}}
    text = json.dumps(metadata, indent=1, sort_keys=True, allow_nan=False) + '\n'
    meta_sha = write_new(args.metadata, text)
    print(json.dumps({**result, 'submission': str(out), 'submission_sha256': sha(out),
                      'metadata_sha256': meta_sha, 'platt': metadata['platt']}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    manifest = sub.add_parser('manifest')
    manifest.add_argument('--data-root', required=True)
    manifest.add_argument('--train-identity', required=True)
    manifest.add_argument('--output', required=True)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--export', required=True)
    evaluate.add_argument('--fold', type=int, required=True)
    evaluate.add_argument('--data-root', required=True)
    evaluate.add_argument('--manifest', required=True)
    evaluate.add_argument('--device', required=True)
    evaluate.add_argument('--csv-path', required=True)
    evaluate.add_argument('--receipt', required=True)
    filt = sub.add_parser('filter')
    filt.add_argument('--folds', required=True)
    filt.add_argument('--fold', type=int, required=True)
    filt.add_argument('--full', required=True)
    filt.add_argument('--output', required=True)
    filt.add_argument('--receipt', required=True)
    test = sub.add_parser('test')
    test.add_argument('--export', required=True)
    test.add_argument('--fold', type=int, required=True)
    test.add_argument('--data-root', required=True)
    test.add_argument('--device', required=True)
    test.add_argument('--csv-path', required=True)
    test.add_argument('--logits-path', required=True)
    test.add_argument('--receipt', required=True)
    fuse = sub.add_parser('fuse')
    fuse.add_argument('--folds', required=True)
    fuse.add_argument('--oof', nargs=3, required=True)
    fuse.add_argument('--test', nargs=3, required=True)
    fuse.add_argument('--exports', nargs=3, required=True)
    fuse.add_argument('--reference', required=True)
    fuse.add_argument('--submit', required=True)
    fuse.add_argument('--metadata', required=True)
    fuse.add_argument('--receipt', required=True)
    args = parser.parse_args()
    {'manifest': cmd_manifest, 'evaluate': cmd_evaluate, 'filter': cmd_filter, 'test': cmd_test,
     'fuse': cmd_fuse}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
