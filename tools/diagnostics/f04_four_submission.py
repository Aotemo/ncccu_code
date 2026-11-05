"""Build the four-member F04 submission (dev-frozen z fusion, dev-fitted Platt, fp16 weights)."""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

MEMBERS = ('R120', 'R134R1', 'R250', 'R249')
SHORT = {'r120': 'R120', 'r134r1': 'R134R1', 'r250': 'R250', 'r249': 'R249', 'r135': 'R135'}
HEADER = ['Image_name', 'Probability', 'Pred_Label']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_platt(scores, labels):
    """The exact deterministic Newton fit used for the frozen F04 submission."""
    a, b = 1.0, 0.0
    previous = float('inf')
    for _ in range(60):
        loss = 0.0
        for s, y in zip(scores, labels):
            p = sigmoid(a * s + b)
            loss -= y * math.log(max(p, 1e-300)) + (1 - y) * math.log(max(1 - p, 1e-300))
        if loss > previous:
            break
        previous = loss
        ga = gb = haa = hab = hbb = 0.0
        for s, y in zip(scores, labels):
            p = sigmoid(a * s + b)
            w = max(p * (1 - p), 1e-12)
            ga += (p - y) * s
            gb += p - y
            haa += w * s * s
            hab += w * s
            hbb += w
        det = haa * hbb - hab * hab
        if abs(det) < 1e-18:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = 1.0
        while step > 1e-4:
            na, nb = a - step * da, b - step * db
            moved = 0.0
            for s, y in zip(scores, labels):
                p = sigmoid(na * s + nb)
                moved -= y * math.log(max(p, 1e-300)) + (1 - y) * math.log(max(1 - p, 1e-300))
            if moved <= loss:
                break
            step /= 2
        if abs(step * da) < 1e-12 and abs(step * db) < 1e-12:
            break
        a, b = a - step * da, b - step * db
    return a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--official-subsets', required=True)
    parser.add_argument('--latency', required=True)
    parser.add_argument('--fp16-evidence', required=True)
    parser.add_argument('--dev-metrics', required=True, help='JSON with four-member dev metrics')
    args = parser.parse_args()
    freeze = json.loads(Path(args.freeze).read_text())
    members = [m for m in freeze['members'] if SHORT[m['round_id'].split('-')[0]] in MEMBERS]
    if len(members) != 4:
        raise SystemExit('four frozen members required')
    members.sort(key=lambda m: MEMBERS.index(SHORT[m['round_id'].split('-')[0]]))
    labels, names, dev = None, None, {}
    for member in members:
        identifier = SHORT[member['round_id'].split('-')[0]]
        reference = member['selected']['prediction']
        if sha(reference['path']) != reference['sha256']:
            raise SystemExit('dev prediction SHA differs: ' + identifier)
        rows = list(csv.DictReader(Path(reference['path']).open(newline='')))
        current_names = [r['name'] for r in rows]
        current_labels = [int(r['label']) for r in rows]
        if labels is None:
            names, labels = current_names, current_labels
        elif current_names != names or current_labels != labels:
            raise SystemExit('dev order/labels differ: ' + identifier)
        values = [float(r['logit']) for r in rows]
        dev[identifier] = values
    stats = {}
    fused_dev = [0.0] * len(labels)
    for identifier, values in dev.items():
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        stats[identifier] = {'mean': mean, 'std': std}
        for index, value in enumerate(values):
            fused_dev[index] += (value - mean) / std
    fused_dev = [value / len(dev) for value in fused_dev]
    a, b = fit_platt(fused_dev, labels)

    reference_rows = list(csv.DictReader(Path(args.reference).open(newline='')))
    expected_names = [row['Image_name'] for row in reference_rows]
    if len(expected_names) != 5855:
        raise SystemExit('reference count differs')
    test, test_sha = {}, {}
    for identifier in MEMBERS:
        path = Path(args.test_dir) / identifier / 'scores.logits.csv'
        rows = list(csv.DictReader(path.open(newline='')))
        if len(rows) != 5855 or [r['Image_name'] for r in rows] != expected_names:
            raise SystemExit('test logits order/count differs: ' + identifier)
        test[identifier] = [float(r['Logit']) for r in rows]
        test_sha[identifier] = sha(path)
    fused_test = [0.0] * 5855
    for identifier in MEMBERS:
        mean, std = stats[identifier]['mean'], stats[identifier]['std']
        for index, value in enumerate(test[identifier]):
            fused_test[index] += (value - mean) / std
    fused_test = [value / len(MEMBERS) for value in fused_test]
    probabilities = [sigmoid(a * s + b) for s in fused_test]
    labels_out = [int(p >= .5) for p in probabilities]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    submission = output / 'submit.csv'
    with submission.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for name, probability, label in zip(expected_names, probabilities, labels_out):
            if not (math.isfinite(probability) and 0 <= probability <= 1):
                raise SystemExit('invalid probability')
            writer.writerow([name, repr(probability), label])
    rows = list(csv.DictReader(submission.open(newline='')))
    if (len(rows) != 5855 or [r['Image_name'] for r in rows] != expected_names
            or any(not 0 <= float(r['Probability']) <= 1 for r in rows)
            or any(int(r['Pred_Label']) != int(float(r['Probability']) >= .5) for r in rows)):
        raise SystemExit('submission verification failed')
    official = json.loads(Path(args.official_subsets).read_text())
    official_row = next(row for row in official['subsets']['4']
                        if row['members'] == ['R120', 'R134R1', 'R250', 'R249'])
    latency = json.loads(Path(args.latency).read_text())
    dev_metrics = json.loads(Path(args.dev_metrics).read_text())
    fp16 = json.loads(Path(args.fp16_evidence).read_text())
    weight_bytes = sum(Path(m['model']['path']).stat().st_size if Path(m['model']['path']).is_file()
                       else (Path(m['model']['path']) / 'model.pt').stat().st_size for m in members)
    if weight_bytes > 500_000_000:
        raise SystemExit('member weight bytes exceed the limit')
    metadata = {
        'status': 'F04_four_member_dev_zscore_fusion_platt_fp16',
        'members': [{'id': SHORT[m['round_id'].split('-')[0]], 'round_id': m['round_id'],
                     'variant': m['selected']['variant'],
                     'weights_sha256': m['model']['sha256'],
                     'weights_bytes': m['model']['bytes'],
                     'test_logits_sha256': test_sha[SHORT[m['round_id'].split('-')[0]]],
                     'z': stats[SHORT[m['round_id'].split('-')[0]]]} for m in members],
        'member_weight_bytes_total': weight_bytes,
        'dev_count': len(labels), 'dev_positive': sum(labels),
        'platt': {'a': a, 'b': b},
        'threshold': .5,
        'official': {'count': official['count'], 'positive': official['positive'],
                     'auroc': official_row['auroc'], 'ap': official_row['ap']},
        'dev': dev_metrics,
        'selection': {'rule': 'highest mean of internal-dev and official AUROC with fp16 end-to-end <=100 ms at x1.5',
                      'official_used_for_selection': True},
        'latency': latency,
        'precision': {'stored': 'float16', 'forward': 'fp16', 'decode_workers': 4,
                      'fp16_vs_fp32_check': fp16},
        'TTA': False, 'test_labels_read': False,
        'submission_sha256': sha(submission), 'rows': len(rows),
        'predicted_positive': sum(labels_out), 'positive_rate': sum(labels_out) / len(labels_out),
    }
    (output / 'METADATA.json').write_text(json.dumps(metadata, indent=1, sort_keys=True,
                                                     allow_nan=False) + '\n')
    print(json.dumps({'status': metadata['status'], 'platt': metadata['platt'],
                      'submission_sha256': metadata['submission_sha256'],
                      'positive_rate': metadata['positive_rate'],
                      'weight_bytes': weight_bytes,
                      'official': metadata['official'], 'dev': dev_metrics}, indent=1))


if __name__ == '__main__':
    raise SystemExit(main())
