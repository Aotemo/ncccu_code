"""P3: fixed internal-dev hard-case diagnostics without model inference.

All row-level scores and image identities stay in the supplied private exp-v3
output.  This tool never reads official data, changes labels, or trains models.
"""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def midrank_quantile(score, population):
    values = sorted(population)
    if not values:
        raise ValueError('nonempty comparison population required')
    left, right = bisect_left(values, score), bisect_right(values, score)
    return (left + (right - left) / 2) / len(values)


def negative_above_mass(score, negatives):
    values = sorted(negatives)
    if not values:
        raise ValueError('nonempty negative population required')
    left, right = bisect_left(values, score), bisect_right(values, score)
    return len(values) - right + (right - left) / 2


def consensus(votes, total):
    if type(votes) is not int or type(total) is not int or total < 1 or not 0 <= votes <= total:
        raise ValueError('invalid model consensus counts')
    return 'all' if votes == total else ('none' if votes == 0 else 'mixed')


def image_hashes(image):
    """64-bit DCT pHash and horizontal dHash of the same prepared image."""
    gray = image.convert('L')
    sample = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    grid = np.arange(32, dtype=np.float64)
    frequencies = np.arange(8, dtype=np.float64)
    cosine = np.cos(np.pi * (2 * grid[:, None] + 1) * frequencies[None, :] / 64)
    coeff = cosine.T @ sample @ cosine
    threshold = float(np.median(coeff.ravel()[1:]))
    phash = 0
    for bit in (coeff.ravel() > threshold):
        phash = (phash << 1) | int(bit)
    pixels = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).ravel():
        dhash = (dhash << 1) | int(bit)
    return phash, dhash


def near_duplicate(first, second):
    return (first[0] ^ second[0]).bit_count() <= 4 and (first[1] ^ second[1]).bit_count() <= 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def ranks(base, source, packet, output):
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19
            or any(os.environ.get(key) != '' for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or not output.resolve().is_relative_to((base / 'P3-HARD-CASE').resolve())):
        raise ValueError('isolated hidden-device Pod CPU output required')
    inventory_path = base / 'F03-AFTER-R246/discover/CANDIDATES.json'
    f03_path = base / 'F03-AFTER-R246/result/RESULT.json'
    if (sha(inventory_path) != 'a7ab1594e9b8a22430e1afb7a439e2559d59c52bbbcf3fd0b3d3c5bfd567dce4'
            or sha(f03_path) != '6dbc0638247d6ac4e0a967528113b2d82834daf87ff3bf49861a5309b95b1119'):
        raise ValueError('predeclared seven-member F03 inputs differ')
    inv, f03 = read(inventory_path), read(f03_path)
    sys.path.insert(0, str(source / 'tools/diagnostics'))
    sys.path.insert(0, str(source / 'runtime/bundle'))
    import paired_evidence as evidence
    reference = evidence._reference(inv['manifest'])
    rows = reference['dev']
    labels = [row['label'] for row in rows]
    if reference['manifest_sha256'] != '7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377' or len(rows) != 5140 or labels.count(1) != 167:
        raise ValueError('predeclared internal dev population differs')
    fixed = inv['fixed_positive_indices']
    if len(fixed) != 20 or len(set(fixed)) != 20 or any(labels[index] != 1 for index in fixed):
        raise ValueError('fixed positive identities differ')
    provenance = read(packet / 'PROVENANCE.json')
    mapping_path = packet / 'PRIVATE-MAPPING.json'
    if (sha(mapping_path) != provenance['private_mapping_sha256']
            or sha(packet / 'BLIND-REVIEW.json') != 'dfaa1d8b8a20b0f097411d9514a9e3d4158200a489274d120ee30f1677c09b85'):
        raise ValueError('blind review or private mapping differs')
    mapping = read(mapping_path)
    unblinded = read(packet / 'UNBLINDED-REVIEW.json')
    by_name = {row['name']: index for index, row in enumerate(rows)}
    if (len(mapping) != 36 or len(unblinded['cases']) != 36
            or unblinded['blind_review_sha256'] != 'dfaa1d8b8a20b0f097411d9514a9e3d4158200a489274d120ee30f1677c09b85'
            or {row['alias'] for row in mapping} != {row['alias'] for row in unblinded['cases']}
            or any(row['label_review'] != 'uncertain' for row in unblinded['cases'])):
        raise ValueError('predeclared 36 uncertain reviews differ')
    review_by_alias = {row['alias']: row for row in unblinded['cases']}
    reviewed = set()
    for item in mapping:
        index = by_name.get(item['name'])
        noted = review_by_alias[item['alias']]
        if (index is None or rows[index]['label'] != item['label'] or rows[index]['sha256'] != item['original_sha256']
                or any(noted[key] != item[key] for key in ('name', 'label', 'original_sha256'))):
            raise ValueError('reviewed row identity differs')
        reviewed.add(index)
    if len(reviewed) != 36 or not set(fixed) <= reviewed:
        raise ValueError('review set or fixed-group overlap differs')
    members = f03['scheme']['members']
    if len(members) != 7 or len(set(members)) != 7:
        raise ValueError('exactly seven frozen members required')
    candidates = {item['round_id']: item for item in inv['members']}
    targets = sorted({i for i, label in enumerate(labels) if label == 1} | reviewed)
    private = {index: {'index': index, 'name': rows[index]['name'], 'label': labels[index],
                       'fixed20': index in fixed, 'reviewed_uncertain': index in reviewed,
                       'members': {}} for index in targets}
    model_receipts = {}
    for member in members:
        row = candidates[member]
        spec = {key: row['spec'][key] for key in ('freeze', 'completion')}
        scores, summary, _ = evidence._selected(spec, reference)
        expected = f03['members'][member]['selected']
        if any(summary[key] != expected[key] for key in ('auroc', 'ap', 'variant', 'cycle', 'successful_updates', 'tensor_sha256')):
            raise ValueError('selected prediction identity or metrics differ from frozen F03')
        negatives = [score for score, label in zip(scores, labels) if label == 0]
        for index in targets:
            score = scores[index]
            above = negative_above_mass(score, negatives)
            private[index]['members'][member] = {'score': score, 'dev_midrank_quantile': midrank_quantile(score, scores),
                'negative_midrank_quantile': midrank_quantile(score, negatives), 'negative_above_mass': above,
                'below_majority_negatives': above > len(negatives) / 2}
        model_receipts[member] = {'freeze_sha256': spec['freeze']['sha256'],
                                  'prediction_sha256': summary['prediction']['sha256'],
                                  'precision_sha256': summary['precision']['sha256'],
                                  'auroc': summary['auroc'], 'ap': summary['ap']}
    for item in private.values():
        item['members_below_majority_negatives'] = sum(row['below_majority_negatives'] for row in item['members'].values())
        item['consensus'] = consensus(item['members_below_majority_negatives'], len(members))
    groups = {'fixed20': [private[i] for i in fixed],
              'remaining_positive': [private[i] for i in targets if labels[i] == 1 and i not in fixed],
              'reviewed_positive': [private[i] for i in sorted(reviewed) if labels[i] == 1],
              'reviewed_negative': [private[i] for i in sorted(reviewed) if labels[i] == 0],
              'all_positive': [private[i] for i in targets if labels[i] == 1]}
    summary = {'status': 'completed_p3_fixed_member_rank_consistency', 'at': time.time(),
        'member_count': len(members), 'population': {'dev': len(rows), 'positive': 167, 'negative': 4973,
            'fixed20': 20, 'reviewed': 36, 'unique_targets': len(targets)},
        'groups': {name: {'count': len(items), 'all': sum(x['consensus'] == 'all' for x in items),
                          'mixed': sum(x['consensus'] == 'mixed' for x in items),
                          'none': sum(x['consensus'] == 'none' for x in items)} for name, items in groups.items()},
        'interpretation': 'All/mixed/none count members below a majority of negatives; negatives are not called mistakes.',
        'input_inventory_sha256': sha(inventory_path), 'f03_sha256': sha(f03_path),
        'manifest_sha256': reference['manifest_sha256'], 'models': model_receipts,
        'official_read': False, 'model_loaded': False, 'new_forward': False, 'GPU_initialized': False}
    output.mkdir(exist_ok=False)
    write(output / 'PRIVATE-RANKS.json', {'rows': list(private.values())})
    write(output / 'SUMMARY-RANKS.json', summary)
    print(json.dumps({'status': summary['status'], 'population': summary['population'], 'groups': summary['groups'],
                      'summary_sha256': sha(output / 'SUMMARY-RANKS.json'),
                      'private_sha256': sha(output / 'PRIVATE-RANKS.json')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ranks(args.base, args.source, args.packet, args.output)


if __name__ == '__main__':
    main()
