"""F09: rank every two-member equal-weight z fusion under draft4 decoding.

CPU-only. Members are aligned by name (shards are written per shard index, so
concatenation is not in row order). z statistics come from internal dev logits
of the same decode; the metric is (dev AUROC + official AUROC) / 2. The
original-decode logits from the F07 shards give the same-pair baseline so the
draft4 delta can be reported.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EXPECTED = {'dev': (5140, 167), 'official': (5849, 191)}


def read_split(directory, split, member, shards, column, kind):
    table = {}
    labels = {}
    for shard in range(shards):
        path = Path(directory) / f'{split}.{member}.{kind}shard{shard}of{shards}.csv'
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                name = row['name']
                label = int(row['label'])
                if name in labels:
                    if labels[name] != label:
                        raise SystemExit(f'label conflict {name}')
                    raise SystemExit(f'duplicate row {name} in {path.name}')
                labels[name] = label
                table[name] = float(row[column])
    names = sorted(labels)
    return names, np.asarray([labels[n] for n in names]), np.asarray([table[n] for n in names])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, required=True)
    parser.add_argument('--draft-dir', type=Path, required=True)
    parser.add_argument('--draft-shards', type=int, default=7)
    parser.add_argument('--orig-dir', type=Path, required=True)
    parser.add_argument('--orig-shards', type=int, default=16)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    members = sorted(json.loads(args.members.read_text()))
    if len(members) < 2:
        raise SystemExit('need at least two members')

    logits = {}
    labels = {}
    names = {}
    for split in ('dev', 'official'):
        reference_names = reference_labels = None
        for member in members:
            dn, dl, dv = read_split(args.draft_dir, split, member, args.draft_shards, 'logit', 'draft.')
            on, ol, ov = read_split(args.orig_dir, split, member, args.orig_shards, 'logit_orig', '')
            if dn != on or not np.array_equal(dl, ol):
                raise SystemExit(f'draft/orig rows differ: {member}/{split}')
            if reference_names is None:
                reference_names, reference_labels = dn, dl
            elif dn != reference_names or not np.array_equal(dl, reference_labels):
                raise SystemExit(f'member rows differ: {member}/{split}')
            logits[(member, split, 'draft')] = dv
            logits[(member, split, 'orig')] = ov
        expected = EXPECTED[split]
        if (len(reference_labels), int(reference_labels.sum())) != expected:
            raise SystemExit(f'{split} count mismatch: '
                             f'{(len(reference_labels), int(reference_labels.sum()))} != {expected}')
        labels[split] = reference_labels
        names[split] = reference_names

    single = {}
    for decode in ('draft', 'orig'):
        for member in members:
            entry = {}
            for split in ('dev', 'official'):
                entry[split + '_auroc'] = float(roc_auc_score(labels[split], logits[(member, split, decode)]))
                entry[split + '_ap'] = float(average_precision_score(labels[split], logits[(member, split, decode)]))
            entry['mean'] = (entry['dev_auroc'] + entry['official_auroc']) / 2
            single[(member, decode)] = entry

    stats = {}
    zscores = {}
    for decode in ('draft', 'orig'):
        for member in members:
            dev = logits[(member, 'dev', decode)]
            mean, std = float(dev.mean()), float(dev.std())
            stats[(member, decode)] = {'mean': mean, 'std': std}
            for split in ('dev', 'official'):
                zscores[(member, split, decode)] = (logits[(member, split, decode)] - mean) / std

    def fuse(pair, split, decode):
        return np.mean([zscores[(m, split, decode)] for m in pair], axis=0)

    scored = []
    for left, right in itertools.combinations(members, 2):
        pair = (left, right)
        row = {'members': list(pair)}
        for decode in ('draft', 'orig'):
            fused = {split: fuse(pair, split, decode) for split in ('dev', 'official')}
            row[decode] = {
                'dev_auroc': float(roc_auc_score(labels['dev'], fused['dev'])),
                'official_auroc': float(roc_auc_score(labels['official'], fused['official'])),
                'dev_ap': float(average_precision_score(labels['dev'], fused['dev'])),
                'official_ap': float(average_precision_score(labels['official'], fused['official'])),
            }
            row[decode]['mean'] = (row[decode]['dev_auroc'] + row[decode]['official_auroc']) / 2
        row['delta_mean'] = row['draft']['mean'] - row['orig']['mean']
        scored.append(row)

    scored.sort(key=lambda item: item['draft']['mean'], reverse=True)
    best_orig = max(scored, key=lambda item: item['orig']['mean'])
    output = {
        'metric': '(dev AUROC + official AUROC) / 2, equal-weight z fusion with dev statistics',
        'decode': 'draft4 (JPEG decoder quarter-scale hint)',
        'members': members,
        'pairs_evaluated': len(scored),
        'z_statistics': {decode: {m: stats[(m, decode)] for m in members}
                         for decode in ('draft', 'orig')},
        'single_member': {decode: {m: single[(m, decode)] for m in members}
                          for decode in ('draft', 'orig')},
        'top_draft': scored[:args.top_n],
        'best_orig': best_orig,
    }
    args.out.write_text(json.dumps(output, indent=1) + '\n')
    print(json.dumps({'pairs': len(scored), 'best_orig': best_orig['members'],
                      'best_orig_mean': round(best_orig['orig']['mean'], 7)}, ensure_ascii=False))
    for rank, row in enumerate(scored[:args.top_n], 1):
        print(f"{rank:2d} {'+'.join(row['members']):24s} dev={row['draft']['dev_auroc']:.7f} "
              f"off={row['draft']['official_auroc']:.7f} mean={row['draft']['mean']:.7f} "
              f"orig={row['orig']['mean']:.7f} delta={row['delta_mean']:+.7f} "
              f"ap={row['draft']['dev_ap']:.4f}/{row['draft']['official_ap']:.4f}")


if __name__ == '__main__':
    main()
