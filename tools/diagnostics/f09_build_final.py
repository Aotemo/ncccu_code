"""F09: build the final two-member inference config from the draft4 selection.

Reads the pair-search result for the dev z statistics, re-reads the draft4 dev
shards of the chosen members to fit Platt scaling on the fused dev score, and
writes the config consumed by predict.py / cost.py.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_dev(directory, member, shards):
    table = {}
    labels = {}
    for shard in range(shards):
        path = Path(directory) / f'dev.{member}.draft.shard{shard}of{shards}.csv'
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                table[row['name']] = float(row['logit'])
                labels[row['name']] = int(row['label'])
    names = sorted(labels)
    return names, np.asarray([labels[n] for n in names]), np.asarray([table[n] for n in names])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--members', type=Path, required=True)
    parser.add_argument('--ids', nargs=2, required=True)
    parser.add_argument('--draft-dir', type=Path, required=True)
    parser.add_argument('--draft-shards', type=int, default=7)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--ages-csv', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    stats = selection['z_statistics']['draft']
    entries = json.loads(args.members.read_text())
    ids = list(args.ids)
    if any(member not in entries for member in ids):
        raise SystemExit('unknown member in --ids')

    fused_parts = []
    reference = None
    members = []
    total_bytes = 0
    for member in ids:
        mean, std = float(stats[member]['mean']), float(stats[member]['std'])
        names, labels, logits = read_dev(args.draft_dir, member, args.draft_shards)
        if reference is None:
            reference = (names, labels)
        elif names != reference[0] or not np.array_equal(labels, reference[1]):
            raise SystemExit('dev rows differ: ' + member)
        fused_parts.append((logits - mean) / std)
        export = Path(entries[member]['export'])
        weights = export / 'model.pt'
        size = weights.stat().st_size
        total_bytes += size
        members.append({
            'id': member,
            'export': str(export),
            'site': entries[member]['site'],
            'site_sha256': entries[member]['site_sha256'],
            'variant': 'ema',
            'model_sha256': sha256(weights),
            'weights_bytes': size,
            'z': {'mean': mean, 'std': std},
        })
    fused = np.mean(np.stack(fused_parts), axis=0)
    reference_labels = reference[1]
    fit = LogisticRegression(max_iter=1000).fit(fused.reshape(-1, 1), reference_labels)
    platt = {'a': float(fit.coef_[0][0]), 'b': float(fit.intercept_[0])}
    auroc = float(roc_auc_score(reference_labels, fused))
    ap = float(average_precision_score(reference_labels, fused))
    config = {
        'schema_version': 1,
        'profile': args.profile,
        'purpose': 'final platform inference for the F09 two-member draft4 submission',
        'decode': 'draft',
        'draft_divisor': 4,
        'precision': 'fp16',
        'TTA': False,
        'platt': platt,
        'threshold': 0.5,
        'ages_csv': args.ages_csv,
        'bundle': args.bundle,
        'member_weight_bytes_total': total_bytes,
        'dev_draft': {'auroc': auroc, 'ap': ap, 'count': int(len(reference_labels)),
                      'positive': int(reference_labels.sum()),
                      'platt_fit': 'sklearn LogisticRegression(max_iter=1000) on fused dev z'},
        'selection': {
            'source': str(args.selection),
            'metric': selection['metric'],
            'rule': 'highest (dev AUROC + official AUROC) / 2 over all two-member equal-weight z fusions',
        },
        'members': members,
    }
    args.out.write_text(json.dumps(config, indent=1) + '\n')
    print(json.dumps({'config': str(args.out), 'members': ids, 'platt': platt,
                      'dev_auroc': auroc, 'dev_ap': ap,
                      'weights_bytes_total': total_bytes}, ensure_ascii=False))


if __name__ == '__main__':
    main()
