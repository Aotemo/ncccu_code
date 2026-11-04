"""Fixed F04 using verified fp32 dev predictions and actual fp16 export sizes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def preregistered_exclusion(round_id):
    # This P2 diagnostic was excluded from the current F04 before its result.
    if round_id == 'r256-small-seed20261111-pairsigmoid-24k-p4-9010-20261104':
        return 'preregistered_P2_variance_diagnostic_not_F04'
    return None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_same_candidate(old, new):
    keys = ('round_id', 'status', 'selected', 'spec', 'science', 'public_source',
            'checkpoint_candidates', 'selected_blob_sha256')
    if any(key not in old or key not in new or old[key] != new[key] for key in keys):
        raise ValueError('candidate changed since the sealed export inventory')


def compare_fixed_sample(fp32, fp16, fp32_dev_std):
    left = np.asarray(fp32, dtype=np.float64)
    right = np.asarray(fp16, dtype=np.float64)
    if (left.shape != (64,) or right.shape != (64,) or not np.isfinite(left).all()
            or not np.isfinite(right).all() or not math.isfinite(fp32_dev_std) or fp32_dev_std <= 0):
        raise ValueError('64 finite paired logits and a positive full-dev fp32 std are required')
    delta = float(np.max(np.abs(left - right)))
    gaps = np.diff(np.sort(left))
    positive = gaps[gaps > 0]
    minimum = float(positive.min()) if len(positive) else None
    ties = int(np.count_nonzero(gaps == 0))
    pairs = np.triu_indices(64, k=1)
    changed = int(np.count_nonzero(np.sign(left[:, None] - left[None, :])[pairs]
                                   != np.sign(right[:, None] - right[None, :])[pairs]))
    return {'count': 64, 'max_abs_logit_difference': delta,
            'max_abs_z_difference': delta / fp32_dev_std,
            'z_scale': 'full_dev_original_fp32_mean_and_population_std_used_for_both',
            'fp32_adjacent_min_positive_gap': minimum,
            'fp32_adjacent_median_gap': float(np.median(gaps)),
            'fp32_adjacent_tie_count': ties, 'changed_pairwise_order_count': changed,
            'small_sample_spacing_passed': bool(minimum is not None and ties == 0
                                                and 10 * delta < minimum and changed == 0)}


def verify_blob(source, metadata, name):
    listing = json.loads((metadata / 'REMOTE-TREE.json').read_text())
    entries = [row for row in listing['tree'] if row.get('type') == 'blob' and row['path'] == name]
    path = source / name
    raw = path.read_bytes()
    digest = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    mode = '100755' if path.stat().st_mode & 0o111 else '100644'
    if path.is_symlink() or len(entries) != 1 or entries[0]['sha'] != digest or entries[0]['mode'] != mode:
        raise ValueError('published Git blob differs: ' + name)
    return sha(path)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'remote-metadata', 'old-inventory', 'inventory', 'export-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('git-commit', 'old-inventory-sha256', 'inventory-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('explicit published source root differs')
    sys.path[:0] = [str(source / 'runtime/bundle'), str(source / 'tools/diagnostics'),
                    str(source / 'tools/pretraining')]
    from source_chain import cpu_context
    from f04_export_member import verify_published
    from f04_size_ensemble import rank_and_cap, equal_zscore
    from ensemble_dev import f03_candidate_gate
    import paired_evidence as pe
    context = cpu_context()
    git_files = verify_published(source, args.remote_metadata, args.git_commit)
    for name in ('tools/diagnostics/f04_bounded_fusion.py', 'tests/test_f04_bounded_fusion.py'):
        git_files[name] = verify_blob(source, args.remote_metadata, name)
    if sha(args.old_inventory) != args.old_inventory_sha256 or sha(args.inventory) != args.inventory_sha256:
        raise ValueError('sealed discovery SHA differs')
    old = json.loads(args.old_inventory.read_text()); inv = json.loads(args.inventory.read_text())
    if (old['status'] != 'read_only_F03_candidate_discovery_no_fusion'
            or inv['status'] != old['status'] or old['official_read'] is not False
            or inv['official_read'] is not False or old['manifest'] != inv['manifest']):
        raise ValueError('sealed internal-dev population differs')
    reference = pe._reference(inv['manifest'])
    labels = [row['label'] for row in reference['dev']]
    if len(labels) != 5140 or sum(labels) != 167:
        raise ValueError('internal-dev population differs')
    work = Path(inv['manifest']['path']).parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(work / 'technical-prep') or output.exists():
        raise ValueError('fresh technical-prep result required')
    cfg = json.loads((source / 'configs/F03-THRESHOLD-EQUAL.json').read_text())
    approved = cfg['approved_general_public_weight_sha256']
    old_rows = {row['round_id']: row for row in old['members']}
    if len(old_rows) != len(old['members']):
        raise ValueError('duplicate old round')
    rows, excluded, by_id, export_refs = [], [], {}, {}
    for order, row in enumerate(inv['members']):
        name = row['round_id']
        fixed_exclusion = preregistered_exclusion(name)
        if fixed_exclusion:
            excluded.append({'round_id': name, 'reason': fixed_exclusion,
                             'selected_auroc': (row.get('selected') or {}).get('auroc')})
            continue
        state, reason = f03_candidate_gate(row, .988, approved)
        if state == 'pending':
            raise ValueError('unresolved candidate: ' + name + ' ' + str(reason))
        if state == 'excluded':
            excluded.append({'round_id': name, 'reason': reason,
                             'selected_auroc': row.get('selected', {}).get('auroc')})
            continue
        if name not in old_rows:
            raise ValueError('new eligible candidate lacks sealed strict export: ' + name)
        require_same_candidate(old_rows[name], row)
        short = name.split('-')[0].upper()
        record_name = 'records/analysis/20261104-f04/' + short + '-EXPORT.json'
        git_files[record_name] = verify_blob(source, args.remote_metadata, record_name)
        published_receipt = source / record_name
        receipt = args.export_root / short / 'EXPORT.json'
        if sha(receipt) != sha(published_receipt):
            raise ValueError('actual export receipt differs from published evidence: ' + name)
        exported = json.loads(receipt.read_text())
        if (exported.get('status') != 'F04_complete_inference_state_exported_and_strictly_reloaded'
                or exported.get('round_id') != name or exported.get('selected') != row['selected']
                or exported.get('inventory', {}).get('sha256') != args.old_inventory_sha256
                or exported.get('selected_blob_sha256') != row['selected_blob_sha256']
                or exported.get('model_constructed') is not True or exported.get('official_read') is not False
                or exported.get('GPU_initialized') is not False or exported['exports'].get('strict_reload') is not True
                or len(row['checkpoint_candidates']) != 1):
            raise ValueError('strict export does not bind selected member: ' + name)
        checkpoint = row['checkpoint_candidates'][0]
        if (not Path(checkpoint['path']).is_file() or sha(checkpoint['sidecar']['path']) != checkpoint['sidecar']['sha256']):
            raise ValueError('selected checkpoint or sidecar unavailable: ' + name)
        sizes = {}
        for precision in ('fp32', 'fp16'):
            item = exported['exports'][precision]; path = Path(item['path'])
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('serialized exported weights differ: ' + name)
            sizes[precision + '_bytes'] = item['bytes']
        rows.append({'id': name, 'order': order, 'auroc': row['selected']['auroc'], **sizes})
        by_id[name] = row
        export_refs[name] = {'path': str(receipt), 'sha256': sha(receipt), 'sizes': sizes}
    half = rank_and_cap(rows, 'fp16')
    full = rank_and_cap(rows, 'fp32')
    if not half['members'] or not full['members']:
        raise ValueError('no member fits the declared capacity')
    output.mkdir(parents=True)
    preflight = {'status': 'F04_fp16_size_members_sealed_before_blend', 'at': time.time(),
                 'inventory_sha256': args.inventory_sha256, 'old_inventory_sha256': args.old_inventory_sha256,
                 'eligible_order': [item['id'] for item in sorted(rows, key=lambda x: (-x['auroc'], x['order']))],
                 'fp16': half, 'fp32': full, 'exports': export_refs, 'excluded': excluded,
                 'git_commit': args.git_commit, 'git_files': git_files, 'official_read': False}
    write(output / 'PREFLIGHT.json', preflight)
    wanted = list(dict.fromkeys(half['members'] + full['members']))
    vectors = {}
    for name in wanted:
        scores, selected, _ = pe._selected(by_id[name]['spec'], reference)
        if selected != by_id[name]['selected']:
            raise ValueError('sealed selected prediction differs: ' + name)
        vectors[name] = scores
    def scheme(selection):
        selected_vectors = {name: vectors[name] for name in selection['members']}
        computed = equal_zscore(labels, selected_vectors, selection['members'])
        return {'members': computed['members'], 'weight_bytes': selection['bytes'],
                'stopped_before': selection['stopped_before'], 'metrics': computed['metrics'],
                'normalization': computed['normalization']}
    report = {'status': 'F04_fp32_dev_predictions_with_fp16_storage_budget', 'at': time.time(),
              'preflight_sha256': sha(output / 'PREFLIGHT.json'), 'inventory_sha256': args.inventory_sha256,
              'population_count': 5140, 'positive_count': 167, 'official_read': False, 'TTA': False,
              'forward_calls': 0, 'fp16_size_fp32_predictions': scheme(half),
              'fp32_size_fp32_predictions_control': scheme(full),
              'limitations': ['No full-dev fp16-stored weight forward has yet been performed.',
                              'The same internal dev was used for member selection; it is not an independent test.']}
    write(output / 'RESULT.json', report)
    print(json.dumps({'status': report['status'], 'preflight_sha256': report['preflight_sha256'],
                      'result_sha256': sha(output / 'RESULT.json'),
                      'main': report['fp16_size_fp32_predictions'],
                      'control': report['fp32_size_fp32_predictions_control']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
