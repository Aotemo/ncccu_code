"""F11b: dev-only z fusion over an arbitrary set of draft4 member shard directories."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def read_dev(directory, member, shards):
    table, labels = {}, {}
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
    parser.add_argument('--spec', action='append', required=True,
                        help='member:directory:shards')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    members, reference, logits = [], None, {}
    for item in args.spec:
        member, directory, shards = item.rsplit(':', 2)
        names, labels, values = read_dev(directory, member, int(shards))
        if reference is None:
            reference = (names, labels)
        elif names != reference[0] or not np.array_equal(labels, reference[1]):
            raise SystemExit('dev rows differ: ' + member)
        members.append(member)
        logits[member] = values

    stats, fused = {}, np.zeros_like(reference[1], dtype=float)
    per_member = {}
    for member in members:
        values = logits[member]
        mean, std = float(values.mean()), float(values.std())
        stats[member] = {'mean': mean, 'std': std}
        fused = fused + (values - mean) / std
        per_member[member] = {'z': {'mean': mean, 'std': std},
                              'dev_auroc': float(roc_auc_score(reference[1], values)),
                              'dev_ap': float(average_precision_score(reference[1], values))}
    fused = fused / len(members)
    labels = reference[1]
    fit = LogisticRegression(max_iter=1000).fit(fused.reshape(-1, 1), labels)
    result = {'members': members, 'dev_rows': int(len(labels)), 'dev_positive': int(labels.sum()),
              'per_member': per_member,
              'fusion_dev_auroc': float(roc_auc_score(labels, fused)),
              'fusion_dev_ap': float(average_precision_score(labels, fused)),
              'platt': {'a': float(fit.coef_[0][0]), 'b': float(fit.intercept_[0])}}
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
