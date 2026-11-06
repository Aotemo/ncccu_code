"""F11: dev-only fusion for the replacement member and the final three-member config."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def read_dev(directory, member, shards, pattern='dev.{m}.draft.shard{s}of{n}.csv'):
    table, labels = {}, {}
    for shard in range(shards):
        path = Path(directory) / pattern.format(m=member, s=shard, n=shards)
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                table[row['name']] = float(row['logit'])
                labels[row['name']] = int(row['label'])
    names = sorted(labels)
    return names, np.asarray([labels[n] for n in names]), np.asarray([table[n] for n in names])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--f11-dir', type=Path, required=True)
    parser.add_argument('--f11-shards', type=int, default=1)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--base-shards', type=int, default=7)
    parser.add_argument('--members', required=True, help='comma separated base members')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    names, labels, f11 = read_dev(args.f11_dir, 'F11', args.f11_shards)
    base = {}
    for member in args.members.split(','):
        other_names, other_labels, values = read_dev(args.base_dir, member, args.base_shards)
        if other_names != names or not np.array_equal(other_labels, labels):
            raise SystemExit('dev rows differ: ' + member)
        base[member] = values

    def stats(values):
        return float(values.mean()), float(values.std())

    mean, std = stats(f11)
    base_mean_std = {m: stats(v) for m, v in base.items()}
    fused = (f11 - mean) / std
    for member, values in base.items():
        m, s = base_mean_std[member]
        fused = fused + (values - m) / s
    fused = fused / (1 + len(base))
    auroc = float(roc_auc_score(labels, fused))
    ap = float(average_precision_score(labels, fused))
    fit = LogisticRegression(max_iter=1000).fit(fused.reshape(-1, 1), labels)
    result = {
        'dev_rows': int(len(labels)), 'dev_positive': int(labels.sum()),
        'f11_member': {'z': {'mean': mean, 'std': std},
                       'dev_auroc': float(roc_auc_score(labels, f11)),
                       'dev_ap': float(average_precision_score(labels, f11))},
        'base_members': {m: {'z': {'mean': base_mean_std[m][0], 'std': base_mean_std[m][1]},
                             'dev_auroc': float(roc_auc_score(labels, v)),
                             'dev_ap': float(average_precision_score(labels, v))}
                         for m, v in base.items()},
        'fusion_dev_auroc': auroc, 'fusion_dev_ap': ap,
        'platt': {'a': float(fit.coef_[0][0]), 'b': float(fit.intercept_[0])},
    }
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
