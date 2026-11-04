"""Describe R255 against frozen F03/F04 dev predictions without selecting models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def correlations(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if (left.ndim != 1 or left.shape != right.shape or len(left) < 2
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or left.std() == 0 or right.std() == 0):
        raise ValueError('finite nonconstant aligned vectors required')
    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        return ((ends - counts + ends + 1) / 2)[inverse]
    return {'Pearson': float(np.corrcoef(left, right)[0, 1]),
            'Spearman': float(np.corrcoef(ranks(left), ranks(right))[0, 1])}


def shared_repair(labels, members, candidate, fixed):
    labels = np.asarray(labels); candidate = np.asarray(candidate, dtype=np.float64)
    arrays = [np.asarray(values, dtype=np.float64) for values in members]
    if (labels.ndim != 1 or set(labels.tolist()) != {0, 1} or len(arrays) != 7
            or candidate.shape != labels.shape or not np.isfinite(candidate).all()
            or any(values.shape != labels.shape or not np.isfinite(values).all() for values in arrays)):
        raise ValueError('seven aligned finite members and a binary candidate are required')
    positives = np.flatnonzero(labels == 1); negatives = np.flatnonzero(labels == 0)
    fixed = set(fixed)
    if not fixed or not fixed <= set(positives.tolist()):
        raise ValueError('fixed positive identities differ')
    result = {name: {'shared': 0, 'repaired': 0, 'ties': 0, 'still_wrong': 0}
              for name in ('fixed20', 'remaining_positive', 'all')}
    for positive in positives:
        shared = np.ones(len(negatives), dtype=bool)
        for values in arrays:
            shared &= values[negatives] > values[positive]
        difference = candidate[positive] - candidate[negatives[shared]]
        counts = {'shared': int(shared.sum()), 'repaired': int((difference > 0).sum()),
                  'ties': int((difference == 0).sum()), 'still_wrong': int((difference < 0).sum())}
        for name in ('all', 'fixed20' if positive in fixed else 'remaining_positive'):
            for key, value in counts.items():
                result[name][key] += value
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pinned(path, expected):
    path = Path(path)
    if path.is_symlink() or sha(path) != expected:
        raise ValueError('pinned input differs: ' + str(path))
    return json.loads(path.read_text())


def verify_own_blob(source, metadata):
    tree = json.loads((metadata / 'REMOTE-TREE.json').read_text())
    name = 'tools/diagnostics/r255_complementarity.py'; path = source / name
    entries = [row for row in tree['tree'] if row['path'] == name]
    raw = path.read_bytes(); blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    mode = '100755' if path.stat().st_mode & 0o111 else '100644'
    if path.is_symlink() or len(entries) != 1 or entries[0]['sha'] != blob or entries[0]['mode'] != mode:
        raise ValueError('published study source differs')
    return sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'remote-metadata', 'f03-inventory', 'f03-result', 'f04-inventory',
                 'f04-preflight', 'r255-safe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--git-commit', required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('published source root differs')
    sys.path[:0] = [str(source / 'tools/pretraining'), str(source / 'runtime/bundle')]
    from source_chain import cpu_context
    from f04_export_member import verify_published
    from f04_size_ensemble import equal_zscore
    import paired_evidence as pe
    context = cpu_context()
    files = verify_published(source, args.remote_metadata, args.git_commit)
    files['tools/diagnostics/r255_complementarity.py'] = verify_own_blob(source, args.remote_metadata)
    f03inv = pinned(args.f03_inventory, 'a7ab1594e9b8a22430e1afb7a439e2559d59c52bbbcf3fd0b3d3c5bfd567dce4')
    f03 = pinned(args.f03_result, '6dbc0638247d6ac4e0a967528113b2d82834daf87ff3bf49861a5309b95b1119')
    f04inv = pinned(args.f04_inventory, '94b9d638e42cb0f2c883af6583c74173c9f45319fd8672fb337e8f74209dd389')
    f04 = pinned(args.f04_preflight, '7839b9d9d6f8ac27c2c990d8c23548e9795567a73a9e88412d47ddbfbf60e5f9')
    safe = pinned(args.r255_safe, '29d27ef6a787ab62e1ff30b150eb413f4da595f31b6530b339c0238382b2cbb7')
    if (safe.get('phase') != 'terminal' or safe.get('safety', {}).get('safe') is not True
            or safe.get('done', {}).get('status') != 'completed' or safe.get('official_read') is not False
            or f03inv['manifest']['sha256'] != '7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377'
            or f04inv['manifest']['sha256'] != f03inv['manifest']['sha256']):
        raise ValueError('R255 safe proof or shared internal-dev manifest differs')
    reference = pe._reference(f03inv['manifest']); rows = reference['dev']
    labels = [row['label'] for row in rows]
    if len(rows) != 5140 or labels.count(1) != 167 or labels.count(0) != 4973:
        raise ValueError('full dev population differs')
    seven = f03['scheme']['members']; five = f04['fp16']['members']
    if len(seven) != 7 or len(set(seven)) != 7 or len(five) != 5 or len(set(five)) != 5:
        raise ValueError('frozen seven/five member sets differ')
    f03rows = {row['round_id']: row for row in f03inv['members']}
    f04rows = {row['round_id']: row for row in f04inv['members']}
    vectors, identities = {}, {}
    for name in dict.fromkeys(seven + five):
        row = f03rows.get(name) or f04rows[name]
        scores, selected, _ = pe._selected(row['spec'], reference)
        if any(selected[key] != row['selected'][key] for key in ('auroc', 'ap', 'variant', 'cycle', 'successful_updates', 'tensor_sha256')):
            raise ValueError('frozen selected identity differs: ' + name)
        if name in seven and any(selected[key] != f03['members'][name]['selected'][key]
                                 for key in ('auroc', 'ap', 'variant', 'cycle', 'successful_updates', 'tensor_sha256')):
            raise ValueError('F03 selected identity differs: ' + name)
        if name in five and name in f04rows and selected['tensor_sha256'] != f04rows[name]['selected']['tensor_sha256']:
            raise ValueError('F04 selected identity differs: ' + name)
        vectors[name] = scores
        identities[name] = {'tensor_sha256': selected['tensor_sha256'], 'prediction_sha256': selected['prediction']['sha256']}
    root = Path(safe['formal_root'])
    freeze_path = root / 'SINGLE-MODEL-FREEZE.json'; completion_path = root / 'evidence/TRAIN_RESULT.json'
    freeze = json.loads(freeze_path.read_text())
    if freeze['selected'] != safe['terminal']['freeze']['selected']:
        raise ValueError('R255 selected differs from complete safe observation')
    spec = {'freeze': {'path': str(freeze_path), 'sha256': sha(freeze_path)},
            'completion': {'path': str(completion_path), 'sha256': sha(completion_path)}}
    candidate, selected, _ = pe._selected(spec, reference)
    if any(selected[key] != safe['selected'][key] for key in ('auroc', 'ap', 'variant', 'cycle', 'successful_updates')):
        raise ValueError('R255 recomputed selected metrics differ')
    repair = shared_repair(labels, [vectors[name] for name in seven], candidate, f03inv['fixed_positive_indices'])
    if [repair[name]['shared'] for name in ('fixed20', 'remaining_positive', 'all')] != [2629, 998, 3627]:
        raise ValueError('frozen P3 shared misordered pairs do not reconstruct')
    compare = {name: correlations(candidate, vectors[name]) for name in five}
    research_name = 'R255-research-only'; order = five + [research_name]
    blend = equal_zscore(labels, {**{name: vectors[name] for name in five}, research_name: candidate}, order)
    output = args.output.resolve(); work = root.parent.parent.resolve(strict=True)
    if output.exists() or not output.is_relative_to(work / 'technical-prep'):
        raise ValueError('fresh technical-prep study output required')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'R255_posthoc_complementarity_description', 'at': time.time(),
              'git_commit': args.git_commit, 'git_files': files, 'execution': context,
              'manifest_sha256': f03inv['manifest']['sha256'], 'population_count': 5140, 'positive_count': 167,
              'f03_inventory_sha256': sha(args.f03_inventory), 'f03_result_sha256': sha(args.f03_result),
              'f04_preflight_sha256': sha(args.f04_preflight), 'r255_safe_sha256': sha(args.r255_safe),
              'r255_selected': selected, 'input_identities': identities,
              'correlations_with_f04': compare, 'shared_pair_repair': repair,
              'research_six_member_equal_z': {'members': order, 'metrics': blend['metrics'],
                  'normalization': blend['normalization'], 'weight_budget_applied': False, 'deployable': False},
              'posthoc_only': True, 'used_for_selection': False, 'official_read': False,
              'model_forward': False, 'TTA': False}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'repair': repair,
                      'research_metrics': blend['metrics'], 'receipt_sha256': sha(output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
