"""F07: assemble sharded TTA logits and search equal-weight z fusions under a forward budget.

CPU-only. Metric = (internal dev AUROC + official AUROC) / 2. Baseline is the
frozen four-member F04 submission mean. Shards for every member follow the same
deterministic row assignment, so shard-order concatenation is consistent across
members; labels and names are cross-checked for every member and split.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASELINE_DEV = 0.9899336657471303
BASELINE_OFFICIAL = 0.9897425505099577
BASELINE_MEAN = (BASELINE_DEV + BASELINE_OFFICIAL) / 2
REQUIRED_MARGIN = 0.0005
R270_COST = 2.25


def read_shards(directory, split, member, shards):
    orig_parts, flip_parts = [], []
    concat_names = []
    concat_labels = []
    for shard in range(shards):
        path = Path(directory) / f'{split}.{member}.shard{shard}of{shards}.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        names = [row['name'] for row in rows]
        labels = [int(row['label']) for row in rows]
        if len(set(names)) != len(names):
            raise SystemExit('duplicate names within shard: ' + path.name)
        concat_names.extend(names)
        concat_labels.extend(labels)
        orig_parts.append(np.asarray([float(row['logit_orig']) for row in rows], dtype=np.float64))
        flip_parts.append(np.asarray([float(row['logit_hflip']) for row in rows], dtype=np.float64))
    values = (np.concatenate(orig_parts), np.concatenate(flip_parts))
    if not np.isfinite(np.vstack(values)).all():
        raise SystemExit('nonfinite logits: ' + member)
    return concat_names, np.asarray(concat_labels), values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, required=True)
    parser.add_argument('--shards-dir', type=Path, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--budget', type=float, default=4.0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    members = json.loads(args.members.read_text())
    ids = sorted(members)
    costs = {mid: float(members[mid].get('cost', R270_COST if mid == 'R270' else 1.0)) for mid in ids}

    logits = {}
    labels = {}
    names = {}
    for split in ('dev', 'official'):
        split_names = split_labels = None
        for mid in ids:
            current_names, current_labels, (orig, flip) = read_shards(args.shards_dir, split, mid, args.shards)
            if split_names is None:
                split_names, split_labels = current_names, current_labels
            elif current_names != split_names or not np.array_equal(current_labels, split_labels):
                raise SystemExit('member row order/labels differ: ' + mid + '/' + split)
            logits[(mid, split)] = (orig, flip)
        labels[split] = split_labels
        names[split] = split_names
        positives = int(split_labels.sum())
        expected = {'dev': (5140, 167), 'official': (5849, 191)}[split]
        if (len(split_labels), positives) != expected:
            raise SystemExit(f'{split} count mismatch: {(len(split_labels), positives)} != {expected}')

    # z statistics on internal dev per member per view configuration.
    z = {}
    dev_stats = {}
    for mid in ids:
        dev_orig, dev_flip = logits[(mid, 'dev')]
        single = dev_orig
        tta = (dev_orig + dev_flip) / 2.0
        dev_stats[mid] = {
            'views1': {'mean': float(single.mean()), 'std': float(single.std())},
            'views2': {'mean': float(tta.mean()), 'std': float(tta.std())},
        }
        if dev_stats[mid]['views1']['std'] <= 0 or dev_stats[mid]['views2']['std'] <= 0:
            raise SystemExit('degenerate dev spread: ' + mid)
        z[mid] = {
            1: {split: (logits[(mid, split)][0] - dev_stats[mid]['views1']['mean']) / dev_stats[mid]['views1']['std']
                for split in ('dev', 'official')},
            2: {split: ((logits[(mid, split)][0] + logits[(mid, split)][1]) / 2 - dev_stats[mid]['views2']['mean'])
                / dev_stats[mid]['views2']['std']
                for split in ('dev', 'official')},
        }

    dev_labels, official_labels = labels['dev'], labels['official']
    results = []
    order = list(range(len(ids)))
    # enumerate subsets by bitmask; view assignment per included member (1 or 2)
    for mask in range(1, 1 << len(ids)):
        chosen = [ids[i] for i in order if mask >> i & 1]
        if sum(costs[c] for c in chosen) > args.budget:
            continue
        for view_bits in range(1 << len(chosen)):
            views = {c: (2 if view_bits >> k & 1 else 1) for k, c in enumerate(chosen)}
            cost = sum(costs[c] * views[c] for c in chosen)
            if cost > args.budget:
                continue
            fused = {
                split: np.mean(np.stack([z[c][views[c]][split] for c in chosen]), axis=0)
                for split in ('dev', 'official')
            }
            dev_auroc = float(roc_auc_score(dev_labels, fused['dev']))
            official_auroc = float(roc_auc_score(official_labels, fused['official']))
            metric = (dev_auroc + official_auroc) / 2
            results.append({
                'members': chosen, 'views': [views[c] for c in chosen], 'cost': round(cost, 3),
                'dev_auroc': dev_auroc, 'official_auroc': official_auroc, 'metric': metric,
                'dev_ap': float(average_precision_score(dev_labels, fused['dev'])),
                'official_ap': float(average_precision_score(official_labels, fused['official'])),
            })
    results.sort(key=lambda row: (-row['metric'], row['cost'], tuple(row['members'])))
    top = results[:10]
    best = results[0]
    decision = {
        'baseline_mean': BASELINE_MEAN,
        'required_metric': BASELINE_MEAN + REQUIRED_MARGIN,
        'best_metric': best['metric'],
        'switch': best['metric'] >= BASELINE_MEAN + REQUIRED_MARGIN,
        'baseline_combo_present': any(sorted(r['members']) == ['R120', 'R134R1', 'R249', 'R250']
                                      and r['views'] == [1, 1, 1, 1] for r in results),
    }
    payload = {'status': 'f07_search_complete', 'members': ids, 'costs': costs,
               'dev_stats': dev_stats, 'top10': top, 'decision': decision,
               'combos_evaluated': len(results)}
    args.out.write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps({'status': payload['status'], 'combos': len(results),
                      'best': {'members': best['members'], 'views': best['views'],
                               'metric': best['metric'], 'dev': best['dev_auroc'],
                               'official': best['official_auroc']},
                      'switch': decision['switch'],
                      'baseline_reproduced': decision['baseline_combo_present']}))


if __name__ == '__main__':
    main()
