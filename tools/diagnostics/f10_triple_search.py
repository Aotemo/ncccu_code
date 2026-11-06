"""F10: rank every three-member equal-weight z fusion over the stored draft4 forwards.

CPU-only. The 14 F09 members come from the seven-shard draft4 directory; the HP
member comes from its single-shard draft4 forward. Members are aligned by name,
z statistics come from internal dev, and the metric is (dev AUROC + official
AUROC) / 2.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EXPECTED = {'dev': (5140, 167), 'official': (5849, 191)}


def read_split(directory, split, member, shards):
    table = {}
    labels = {}
    for shard in range(shards):
        path = Path(directory) / f'{split}.{member}.draft.shard{shard}of{shards}.csv'
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                name = row['name']
                label = int(row['label'])
                if name in labels:
                    raise SystemExit(f'duplicate row {name} in {path.name}')
                labels[name] = label
                table[name] = float(row['logit'])
    names = sorted(labels)
    return names, np.asarray([labels[n] for n in names]), np.asarray([table[n] for n in names])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, required=True)
    parser.add_argument('--draft-dir', type=Path, required=True)
    parser.add_argument('--draft-shards', type=int, default=7)
    parser.add_argument('--extra-members', type=Path, default=None)
    parser.add_argument('--extra-dir', type=Path, default=None)
    parser.add_argument('--extra-shards', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    members = sorted(json.loads(args.members.read_text()))
    extra = sorted(json.loads(args.extra_members.read_text())) if args.extra_members else []
    if set(members) & set(extra):
        raise SystemExit('member id list collides between the two sources')

    logits = {}
    labels = {}
    for split in ('dev', 'official'):
        reference = None
        for member in members + extra:
            if member in extra:
                names, current, values = read_split(args.extra_dir, split, member, args.extra_shards)
            else:
                names, current, values = read_split(args.draft_dir, split, member, args.draft_shards)
            if reference is None:
                reference = (names, current)
            elif names != reference[0] or not np.array_equal(current, reference[1]):
                raise SystemExit(f'rows differ: {member}/{split}')
            logits[(member, split)] = values
        expected = EXPECTED[split]
        if (len(reference[1]), int(reference[1].sum())) != expected:
            raise SystemExit(f'{split} count mismatch')
        labels[split] = reference[1]

    all_members = members + extra
    z = {}
    single = {}
    for member in all_members:
        dev = logits[(member, 'dev')]
        mean, std = float(dev.mean()), float(dev.std())
        z[member] = {split: (logits[(member, split)] - mean) / std for split in ('dev', 'official')}
        single[member] = {'z': {'mean': mean, 'std': std}}
        for split in ('dev', 'official'):
            single[member][split] = {
                'auroc': float(roc_auc_score(labels[split], logits[(member, split)])),
                'ap': float(average_precision_score(labels[split], logits[(member, split)])),
            }

    scored = []
    for combo in itertools.combinations(all_members, 3):
        row = {'members': list(combo)}
        for split in ('dev', 'official'):
            fused = np.mean([z[m][split] for m in combo], axis=0)
            row[split] = {'auroc': float(roc_auc_score(labels[split], fused)),
                          'ap': float(average_precision_score(labels[split], fused))}
        row['mean_auroc'] = (row['dev']['auroc'] + row['official']['auroc']) / 2
        scored.append(row)
    scored.sort(key=lambda item: item['mean_auroc'], reverse=True)

    output = {
        'metric': '(dev AUROC + official AUROC) / 2, equal-weight z fusion with dev statistics',
        'decode': 'draft4',
        'members': all_members,
        'combos_evaluated': len(scored),
        'single_member': single,
        'top': scored[:args.top_n],
    }
    args.out.write_text(json.dumps(output, indent=1) + '\n')
    print(json.dumps({'combos': len(scored)}))
    for rank, row in enumerate(scored[:args.top_n], 1):
        print(f"{rank:2d} {'+'.join(row['members']):42s} dev={row['dev']['auroc']:.7f} "
              f"off={row['official']['auroc']:.7f} mean={row['mean_auroc']:.7f} "
              f"ap={row['dev']['ap']:.4f}/{row['official']['ap']:.4f}")


if __name__ == '__main__':
    main()
