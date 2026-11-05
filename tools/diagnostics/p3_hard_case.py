"""P3: fixed internal-dev hard-case diagnostics without model inference.

All row-level scores and image identities stay in the supplied private exp-v3
output.  This tool never reads official data, changes labels, or trains models.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


class HashIndex:
    """Exact candidate generation: <=4 bit changes leave one of five blocks unchanged."""
    def __init__(self, entries):
        self.entries = dict(entries)
        if len(self.entries) != len(entries):
            raise ValueError('hash index identities must be unique')
        self.buckets = defaultdict(set)
        for key, (phash, _) in self.entries.items():
            for block, shift in enumerate((0, 13, 26, 39, 52)):
                width = 12 if block == 4 else 13
                self.buckets[(block, (phash >> shift) & ((1 << width) - 1))].add(key)

    def find(self, query):
        candidates = set()
        for block, shift in enumerate((0, 13, 26, 39, 52)):
            width = 12 if block == 4 else 13
            candidates.update(self.buckets[(block, (query[0] >> shift) & ((1 << width) - 1))])
        return sorted(key for key in candidates if near_duplicate(query, self.entries[key]))


def quality_thresholds(reference, required=147):
    keys = ('brightness', 'contrast', 'sharpness', 'field')
    if len(reference) != required or any(row is None or any(k not in row for k in keys) for row in reference):
        return None
    values = {key: [float(row[key]) for row in reference] for key in keys}
    if any(not np.isfinite(value) for row in values.values() for value in row):
        return None
    return {'brightness_low': float(np.quantile(values['brightness'], .05)),
            'brightness_high': float(np.quantile(values['brightness'], .95)),
            **{key + '_low': float(np.quantile(values[key], .05)) for key in keys[1:]}}


def quality_extreme(features, thresholds):
    if features is None or thresholds is None:
        return None
    flags = [features['brightness'] < thresholds['brightness_low'] or features['brightness'] > thresholds['brightness_high']]
    flags.extend(features[key] < thresholds[key + '_low'] for key in ('contrast', 'sharpness', 'field'))
    return sum(flags) >= 2


def image_scan_order(fit, dev, fit_labels, priority_dev):
    if (len(set(fit)) != len(fit) or len(set(dev)) != len(dev) or set(fit) & set(dev)
            or len(set(priority_dev)) != len(priority_dev)
            or any(type(i) is not int or not 0 <= i < len(dev) for i in priority_dev)
            or any(fit_labels.get(i) not in (0, 1) for i in fit)):
        raise ValueError('invalid disjoint sealed image populations')
    chosen = set(priority_dev)
    return ([('dev', i, dev[i]) for i in sorted(chosen)]
            + [('fit', None, i) for i in sorted(fit) if fit_labels[i] == 1]
            + [('dev', i, value) for i, value in enumerate(dev) if i not in chosen]
            + [('fit', None, i) for i in sorted(fit) if fit_labels[i] == 0])


def p3_next_workers(workers, ceiling, latency_p95, d_ratio, low_streak):
    """Apply the preregistered NFS feedback policy to the next batch."""
    if (type(workers) is not int or type(ceiling) is not int or not 1 <= workers <= ceiling
            or type(low_streak) is not int or low_streak < 0
            or not np.isfinite(latency_p95) or latency_p95 < 0
            or not np.isfinite(d_ratio) or not 0 <= d_ratio <= 1):
        raise ValueError('invalid adaptive scan feedback')
    pause = latency_p95 > 5 or d_ratio >= .5
    if latency_p95 > 2 or d_ratio >= .25:
        return max(1, workers // 2), 0, pause
    if latency_p95 < .25 and d_ratio < .1:
        low_streak += 1
        if low_streak >= 2:
            return min(ceiling, workers + 8), 0, pause
        return workers, low_streak, pause
    return workers, 0, pause


def p3_d_ratio(threads):
    """Read only the scanner's live native worker states from procfs."""
    states = []
    for thread in threads:
        if not thread.is_alive() or thread.native_id is None:
            continue
        try:
            value = (Path('/proc/self/task') / str(thread.native_id) / 'stat').read_text()
        except FileNotFoundError:
            continue
        states.append(value.rsplit(')', 1)[1].split()[0])
    return sum(state == 'D' for state in states) / len(states) if states else 0.0


def p3_parallel_batch(items, worker, workers, d_sampler=p3_d_ratio):
    """Bound submissions, stop adding work on acute D pressure, retain order."""
    if not items or len(items) > 64 or type(workers) is not int or not 1 <= workers:
        raise ValueError('one batch of at most 64 items and a positive worker count required')
    completed, pending, latencies = {}, {}, []
    next_index, peak_d, emergency = 0, 0.0, False
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='p3-scan') as pool:
        while pending or (next_index < len(items) and not emergency):
            while not emergency and next_index < len(items) and len(pending) < workers:
                future = pool.submit(worker, items[next_index])
                pending[future] = next_index
                next_index += 1
            ratio = d_sampler(pool._threads)
            if not np.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError('invalid scanner worker D-state sample')
            peak_d = max(peak_d, ratio)
            emergency |= ratio >= .5
            done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
            for future in done:
                value, latency = future.result()
                if not np.isfinite(latency) or latency < 0:
                    raise ValueError('invalid cache read latency')
                completed[pending.pop(future)] = value
                latencies.append(latency)
    if set(completed) != set(range(next_index)) or not completed:
        raise ValueError('incomplete adaptive scan batch')
    return [completed[index] for index in range(next_index)], latencies, peak_d


def shared_negative_indices(positive_scores, negative_scores):
    if not positive_scores or len(positive_scores) != len(negative_scores):
        raise ValueError('one negative vector for each model is required')
    if len({len(values) for values in negative_scores}) != 1:
        raise ValueError('negative identities must have the same order across models')
    shared = None
    for score, negatives in zip(positive_scores, negative_scores):
        above = {index for index, negative in enumerate(negatives) if negative > score}
        shared = above if shared is None else shared & above
    return shared


def shared_negative_count(positive_scores, negative_scores):
    return len(shared_negative_indices(positive_scores, negative_scores))


def shared_negative_support(shared_by_positive):
    return dict(Counter(index for indices in shared_by_positive for index in indices))


def shared_pair_summary(shared_by_positive):
    support = shared_negative_support(shared_by_positive)
    return {'pair_count': sum(support.values()), 'unique_negative_count': len(support),
            'negative_count_support_at_least_three': sum(count >= 3 for count in support.values())}


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
    model_receipts, negatives_by_member = {}, {}
    for member in members:
        row = candidates[member]
        spec = {key: row['spec'][key] for key in ('freeze', 'completion')}
        scores, summary, _ = evidence._selected(spec, reference)
        expected = f03['members'][member]['selected']
        if any(summary[key] != expected[key] for key in ('auroc', 'ap', 'variant', 'cycle', 'successful_updates', 'tensor_sha256')):
            raise ValueError('selected prediction identity or metrics differ from frozen F03')
        negatives = [score for score, label in zip(scores, labels) if label == 0]
        negatives_by_member[member] = negatives
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
    shared_by_positive = {}
    for index, item in private.items():
        item['members_below_majority_negatives'] = sum(row['below_majority_negatives'] for row in item['members'].values())
        item['consensus'] = consensus(item['members_below_majority_negatives'], len(members))
        if item['label'] == 1:
            item['members_with_any_misordered_negative'] = sum(row['negative_above_mass'] > 0 for row in item['members'].values())
            item['any_misorder_consensus'] = consensus(item['members_with_any_misordered_negative'], len(members))
            shared_by_positive[index] = shared_negative_indices(
                [item['members'][member]['score'] for member in members],
                [negatives_by_member[member] for member in members])
            item['shared_strictly_misordered_negatives'] = len(shared_by_positive[index])
    groups = {'fixed20': [private[i] for i in fixed],
              'remaining_positive': [private[i] for i in targets if labels[i] == 1 and i not in fixed],
              'reviewed_positive': [private[i] for i in sorted(reviewed) if labels[i] == 1],
              'reviewed_negative': [private[i] for i in sorted(reviewed) if labels[i] == 0],
              'all_positive': [private[i] for i in targets if labels[i] == 1]}
    def group_summary(items):
        result = {'count': len(items), 'all': sum(x['consensus'] == 'all' for x in items),
                  'mixed': sum(x['consensus'] == 'mixed' for x in items),
                  'none': sum(x['consensus'] == 'none' for x in items),
                  'negative_percentile_median_by_member': {member: float(np.median(
                      [x['members'][member]['negative_midrank_quantile'] for x in items])) for member in members}}
        if items and all(x['label'] == 1 for x in items):
            shared = [x['shared_strictly_misordered_negatives'] for x in items]
            result.update({'any_misorder_all': sum(x['any_misorder_consensus'] == 'all' for x in items),
                           'any_misorder_mixed': sum(x['any_misorder_consensus'] == 'mixed' for x in items),
                           'any_misorder_none': sum(x['any_misorder_consensus'] == 'none' for x in items),
                           'shared_strict_negative_positive_count': sum(x > 0 for x in shared),
                           'shared_strict_negative_count_median': float(np.median(shared)),
                           'shared_strict_negative_count_max': max(shared),
                           'shared_strict_pair_summary': shared_pair_summary(
                               [shared_by_positive[x['index']] for x in items])})
        return result

    group_results = {name: group_summary(items) for name, items in groups.items()}
    pair_denominator = 167 * 4973
    pair_total = group_results['all_positive']['shared_strict_pair_summary']['pair_count']
    if (pair_denominator != 830491 or pair_total != sum(group_results[name]['shared_strict_pair_summary']['pair_count']
                                                         for name in ('fixed20', 'remaining_positive'))):
        raise ValueError('fixed groups or positive-negative denominator differ')
    summary = {'status': 'completed_p3_fixed_member_rank_consistency', 'at': time.time(),
        'member_count': len(members), 'population': {'dev': len(rows), 'positive': 167, 'negative': 4973,
            'fixed20': 20, 'reviewed': 36, 'unique_targets': len(targets)},
        'groups': group_results,
        'shared_strict_pair_denominator': pair_denominator,
        'nonnegative_seven_member_auroc_upper_bound': 1 - pair_total / pair_denominator,
        'upper_bound_scope': 'positive member standardization and nonnegative ensemble weights only; no new members',
        'interpretation': 'All/mixed/none count members below a majority of negatives; negatives are not called mistakes.',
        'input_inventory_sha256': sha(inventory_path), 'f03_sha256': sha(f03_path),
        'manifest_sha256': reference['manifest_sha256'], 'models': model_receipts,
        'official_read': False, 'model_loaded': False, 'new_forward': False, 'GPU_initialized': False}
    fixed_support = shared_negative_support([shared_by_positive[index] for index in fixed])
    negative_dev_indices = [index for index, label in enumerate(labels) if label == 0]
    if (len(negative_dev_indices) != 4973 or len(fixed_support) != 603
            or sum(count >= 3 for count in fixed_support.values()) != 241):
        raise ValueError('reconstructed fixed20 shared-negative population differs')
    private_negatives = [{'dev_index': negative_dev_indices[index], 'fixed20_support': count,
                          'support_at_least_three': count >= 3}
                         for index, count in sorted(fixed_support.items())]
    output.mkdir(exist_ok=False)
    write(output / 'PRIVATE-RANKS.json', {'rows': list(private.values())})
    write(output / 'PRIVATE-SHARED-NEGATIVES.json', {'rows': private_negatives,
          'manifest_sha256': reference['manifest_sha256'], 'f03_sha256': sha(f03_path),
          'definition': 'negative exceeds fixed positive under all seven members'})
    summary['private_shared_negative_sha256'] = sha(output / 'PRIVATE-SHARED-NEGATIVES.json')
    write(output / 'SUMMARY-RANKS.json', summary)
    compact = {name: {key: value for key, value in summary['groups'][name].items()
                      if key != 'negative_percentile_median_by_member'}
               for name in ('fixed20', 'remaining_positive', 'reviewed_negative')}
    print(json.dumps({'status': summary['status'], 'population': summary['population'], 'groups': compact,
                      'summary_sha256': sha(output / 'SUMMARY-RANKS.json'),
                      'private_sha256': sha(output / 'PRIVATE-RANKS.json'),
                      'private_shared_negative_sha256': summary['private_shared_negative_sha256']}, indent=2), flush=True)


def image_inputs(base, site_path, site_sha):
    """Resolve only previously sealed fit/dev populations and fixed F03 identities."""
    if sha(site_path) != site_sha:
        raise ValueError('Pod site binding SHA differs')
    site = read(site_path)
    if site['pod']['uid'] != os.environ['NCCCU_POD_UID']:
        raise ValueError('current Pod UID differs')
    manifest_path = Path(site['metadata_root']) / 'manifest.json'
    inv_path = base / 'F03-AFTER-R246/discover/CANDIDATES.json'
    shared_path = base / 'P3-HARD-CASE/RANKS-v4/PRIVATE-SHARED-NEGATIVES.json'
    ranks_path = base / 'P3-HARD-CASE/RANKS-v4/PRIVATE-RANKS.json'
    expected = ((manifest_path, '7cc0f8b785d7f4b578839eadcca5d45ac724451eea75ed3c8af7332eb9409377'),
                (inv_path, 'a7ab1594e9b8a22430e1afb7a439e2559d59c52bbbcf3fd0b3d3c5bfd567dce4'),
                (shared_path, 'e040f882865484cacb38350661ce8d4848bd7d9ba1b02e72adb827df4c894667'))
    if any(sha(path) != digest for path, digest in expected):
        raise ValueError('sealed manifest or fixed F03 identities differ')
    manifest, inv, shared, ranks = map(read, (manifest_path, inv_path, shared_path, ranks_path))
    fit, dev, rows = manifest['fit_indices'], manifest['dev_indices'], manifest['rows']
    if len(fit) != 46256 or len(dev) != 5140 or sum(rows[i]['label'] == 1 for i in dev) != 167:
        raise ValueError('sealed population counts differ')
    fixed = inv['fixed_positive_indices']
    negatives = shared['rows']
    if (len(fixed) != 20 or len(set(fixed)) != 20 or any(rows[dev[i]]['label'] != 1 for i in fixed)
            or len(negatives) != 603 or len({x['dev_index'] for x in negatives}) != 603
            or sum(x['support_at_least_three'] for x in negatives) != 241
            or shared['manifest_sha256'] != sha(manifest_path)
            or shared['f03_sha256'] != '6dbc0638247d6ac4e0a967528113b2d82834daf87ff3bf49861a5309b95b1119'
            or any(rows[dev[x['dev_index']]]['label'] != 0 for x in negatives)):
        raise ValueError('fixed 20/603/241 groups differ')
    ranked = ranks['rows']
    if (len({x['index'] for x in ranked}) != len(ranked)
            or any(rows[dev[x['index']]]['name'] != x['name'] or rows[dev[x['index']]]['label'] != x['label'] for x in ranked)
            or sum(x['reviewed_uncertain'] for x in ranked) != 36):
        raise ValueError('fixed technical reviews differ')
    priority = {i for i, global_i in enumerate(dev) if rows[global_i]['label'] == 1}
    priority.update(x['index'] for x in ranked if x['reviewed_uncertain'])
    priority.update(x['dev_index'] for x in negatives)
    order = image_scan_order(fit, dev, {i: rows[i]['label'] for i in fit}, sorted(priority))
    return {'site': site, 'manifest': manifest, 'inventory': inv, 'shared': shared, 'ranked': ranked,
            'priority': sorted(priority), 'order': order,
            'identities': {'manifest_sha256': sha(manifest_path), 'inventory_sha256': sha(inv_path),
                           'shared_negative_sha256': sha(shared_path), 'ranked_sha256': sha(ranks_path),
                           'site_sha256': site_sha}}


def image_init(base, source, output, site, site_sha):
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(k) != '' for k in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or not output.resolve().is_relative_to((base / 'P3-HARD-CASE').resolve())):
        raise ValueError('isolated hidden-device P3 output required')
    inputs = image_inputs(base, site, site_sha)
    output.mkdir(exist_ok=False)
    (output / 'parts').mkdir()
    result = {'status': 'sealed_p3_image_inputs', 'at': time.time(), **inputs['identities'],
              'source_sha256': sha(source / 'tools/diagnostics/p3_hard_case.py'),
              'fit_count': 46256, 'dev_count': 5140, 'priority_dev_count': len(inputs['priority']),
              'order': 'priority dev, fit positive, remaining dev, fit negative',
              'max_images_per_second': 4, 'unknown_never_replaced': True,
              'official_read': False, 'GPU_initialized': False, 'forward_calls': 0}
    write(output / 'INPUTS.json', result)
    print(json.dumps({**result, 'receipt_sha256': sha(output / 'INPUTS.json')}), flush=True)


def image_parts(output, order):
    entries = []
    for path in sorted((output / 'parts').glob('part-*.json')):
        part = read(path)
        if part['start'] != len(entries) or not part['rows']:
            raise ValueError('image scan part gap or duplicate')
        for row in part['rows']:
            if (row['split'], row['dev_index'], row['global_index']) != order[len(entries)]:
                raise ValueError('image scan order changed')
            entries.append(row)
    return entries


def image_scan_row(item, row, cache, data_root, prepare_image, lexical_path, image_features):
    """The same sealed-image, hash and quality definitions for either scanner."""
    import io
    from PIL import ImageOps
    split, dev_index, global_index = item
    relative, expected = row['relative_path'], row['sha256']
    lexical_path(relative, 'train')
    image, source_kind, error = None, 'unknown', None
    started = time.monotonic()
    try:
        image = cache.read(relative, expected)
        source_kind = 'sealed_readonly_512_cache'
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        try:
            path = data_root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(data_root / 'train'):
                raise ValueError('source path differs')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('original source SHA differs')
            with Image.open(io.BytesIO(raw)) as original:
                image = prepare_image(ImageOps.exif_transpose(original).convert('RGB'), 512)
            source_kind = 'sha_verified_original'
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            error = type(exc).__name__
    read_seconds = time.monotonic() - started
    quality = hashes = None
    if image is not None:
        ratio = 512 / max(image.size)
        target = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
        view = Image.new('RGB', (512, 512))
        view.paste(image.resize(target, Image.Resampling.BILINEAR),
                   (round((512 - target[0]) * .5), round((512 - target[1]) * .5)))
        hashes = [f'{x:016x}' for x in image_hashes(view)]
        if split == 'dev':
            try:
                features = image_features(view)
                quality = {'brightness': features['content_luminance_mean'],
                           'contrast': features['content_luminance_std'],
                           'sharpness': features['green_laplacian_variance'],
                           'field': 1 - features['black_fraction']}
            except ValueError as exc:
                error = type(exc).__name__
    return ({'split': split, 'dev_index': dev_index, 'global_index': global_index,
             'label': row['label'], 'raw_sha256': expected, 'hashes': hashes,
             'quality': quality, 'source': source_kind, 'error': error}, read_seconds)


def image_write_part(output, start, result):
    if not result:
        raise ValueError('cannot publish an empty image scan part')
    path = output / 'parts' / f'part-{start:06d}-{start + len(result) - 1:06d}.json'
    temporary = path.parent / ('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('x') as stream:
        json.dump({'start': start, 'rows': result}, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    return path


def image_scan(base, source, output, site, site_sha, max_images):
    sys.path.insert(0, str(source / 'runtime/bundle'))
    sys.path.insert(0, str(source / 'tools/diagnostics'))
    from retrain.preprocessing import PreparedImageCache, prepare_image
    from retrain.verification import lexical_path
    from persistent_errors import image_features
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(k) != '' for k in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or not output.resolve().is_relative_to((base / 'P3-HARD-CASE').resolve())):
        raise ValueError('isolated hidden-device P3 output required')
    if type(max_images) is not int or not 1 <= max_images <= 600:
        raise ValueError('one bounded scan part of 1 to 600 images required')
    inputs = image_inputs(base, site, site_sha)
    sealed = read(output / 'INPUTS.json')
    if (any(sealed[k] != v for k, v in inputs['identities'].items())
            or sealed['source_sha256'] != sha(source / 'tools/diagnostics/p3_hard_case.py')):
        raise ValueError('sealed image scan identity changed')
    order = inputs['order']
    prior = image_parts(output, order)
    start = len(prior)
    if start == len(order):
        print(json.dumps({'status': 'image_scan_already_complete', 'count': start}), flush=True)
        return
    os.environ['TRAIN_IMAGE_CACHE_READONLY'] = '1'
    data_root = Path(inputs['site']['data_root']).resolve(strict=True)
    cache = PreparedImageCache(data_root, Path(inputs['site']['prepared_cache']) / 'train/sealed', 512)
    rows, result = inputs['manifest']['rows'], []
    started = time.monotonic()
    for ordinal, item in enumerate(order[start:start + max_images]):
        delay = ordinal * .25 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
        scanned, _ = image_scan_row(item, rows[item[2]], cache, data_root,
                                    prepare_image, lexical_path, image_features)
        result.append(scanned)
    path = image_write_part(output, start, result)
    print(json.dumps({'status': 'image_scan_part_complete', 'start': start, 'count': len(result),
                      'completed': start + len(result), 'total': len(order),
                      'unknown': sum(x['hashes'] is None for x in result),
                      'source_kinds': dict(Counter(x['source'] for x in result)),
                      'part_sha256': sha(path), 'official_read': False, 'GPU_initialized': False}), flush=True)


def image_scan_adaptive(base, source, original_source, output, site, site_sha, max_images):
    """Resume sealed P3 parts with bounded workers and NFS feedback."""
    import fcntl
    from threadpoolctl import threadpool_limits
    sys.path.insert(0, str(source / 'runtime/bundle'))
    sys.path.insert(0, str(source / 'tools/diagnostics'))
    from retrain.preprocessing import PreparedImageCache, prepare_image
    from retrain.verification import lexical_path
    from persistent_errors import image_features
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(k) != '' for k in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or not output.resolve().is_relative_to((base / 'P3-HARD-CASE').resolve())):
        raise ValueError('isolated hidden-device P3 output required')
    if type(max_images) is not int or not 1 <= max_images <= 51396:
        raise ValueError('a bounded number of images is required')
    threads = int(os.environ['NCCCU_CPU_THREADS'])
    mask = json.loads(os.environ['NCCCU_CPU_MASK'])
    if threads < 2 or len(mask) != threads or set(mask) != os.sched_getaffinity(0):
        raise ValueError('explicit CPU allocation with at least two cores required')
    inputs = image_inputs(base, site, site_sha)
    sealed = read(output / 'INPUTS.json')
    if (any(sealed[k] != v for k, v in inputs['identities'].items())
            or not original_source.is_absolute() or original_source.is_symlink()
            or sealed['source_sha256'] != sha(original_source)
            or sha(__file__) != sha(source / 'tools/diagnostics/p3_hard_case.py')):
        raise ValueError('sealed input or source identity changed')
    order = inputs['order']
    prior = image_parts(output, order)
    start = len(prior)
    if start == len(order):
        print(json.dumps({'status': 'image_scan_already_complete', 'count': start}), flush=True)
        return
    lock_path = output / '.adaptive-scan.lock'
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck after acquiring the new scanner's lease; old scanner must already be stopped.
        if len(image_parts(output, order)) != start:
            raise ValueError('another scanner changed the sealed parts')
        os.environ['TRAIN_IMAGE_CACHE_READONLY'] = '1'
        data_root = Path(inputs['site']['data_root']).resolve(strict=True)
        cache = PreparedImageCache(data_root, Path(inputs['site']['prepared_cache']) / 'train/sealed', 512)
        rows = inputs['manifest']['rows']
        ceiling = threads - 1
        workers, low_streak, danger_streak = min(16, ceiling), 0, 0
        end = min(start + max_images, len(order))
        unknown = 0
        def scan(item):
            return image_scan_row(item, rows[item[2]], cache, data_root,
                                  prepare_image, lexical_path, image_features)
        with threadpool_limits(limits=1):
            while start < end:
                result, latencies, d_ratio = p3_parallel_batch(order[start:min(end, start + 64)], scan, workers)
                path = image_write_part(output, start, result)
                start += len(result)
                unknown += sum(x['hashes'] is None for x in result)
                p95 = float(np.quantile(latencies, .95))
                next_workers, low_streak, pause = p3_next_workers(workers, ceiling, p95, d_ratio, low_streak)
                danger_streak = danger_streak + 1 if pause else 0
                print(json.dumps({'status': 'adaptive_p3_part_complete', 'completed': start,
                    'total': len(order), 'count': len(result), 'workers': workers, 'next_workers': next_workers,
                    'read_seconds_p95': p95, 'worker_d_ratio_peak': d_ratio,
                    'unknown_so_far_this_run': unknown, 'part_sha256': sha(path),
                    'official_read': False, 'GPU_initialized': False}), flush=True)
                workers = next_workers
                if danger_streak >= 3:
                    raise RuntimeError('NFS pressure persisted for three batches; remaining rows unknown')
                if pause and start < end:
                    time.sleep(30)
        print(json.dumps({'status': 'adaptive_p3_bounded_run_complete', 'completed': start,
                          'total': len(order), 'unknown_this_run': unknown,
                          'official_read': False, 'GPU_initialized': False}), flush=True)
    finally:
        os.close(descriptor)


def image_summary(base, output, site, site_sha):
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(k) != '' for k in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or not output.resolve().is_relative_to((base / 'P3-HARD-CASE').resolve())):
        raise ValueError('isolated hidden-device P3 output required')
    inputs = image_inputs(base, site, site_sha)
    sealed = read(output / 'INPUTS.json')
    if any(sealed[k] != v for k, v in inputs['identities'].items()):
        raise ValueError('image scan inputs changed')
    rows = image_parts(output, inputs['order'])
    count, total = len(rows), len(inputs['order'])
    dev = {x['dev_index']: x for x in rows if x['split'] == 'dev'}
    available = [x for x in rows if x['hashes'] is not None]
    fixed = set(inputs['inventory']['fixed_positive_indices'])
    high = {x['dev_index'] for x in inputs['shared']['rows']}
    high3 = {x['dev_index'] for x in inputs['shared']['rows'] if x['support_at_least_three']}
    queries = [dev[i] for i in inputs['priority'] if i in dev and dev[i]['hashes'] is not None]
    def pair(row):
        return tuple(int(x, 16) for x in row['hashes'])
    targets = [x for x in available if x['split'] == 'fit' or x['label'] == 0 or x['dev_index'] in fixed]
    index = HashIndex([(i, pair(x)) for i, x in enumerate(targets)])
    matches, by_query = [], defaultdict(list)
    for query in queries:
        for target_idx in index.find(pair(query)):
            target = targets[target_idx]
            if target['global_index'] == query['global_index']:
                continue
            item = {'query_dev_index': query['dev_index'], 'target_global_index': target['global_index'],
                    'target_split': target['split'], 'target_dev_index': target['dev_index'],
                    'opposite_label': query['label'] != target['label'],
                    'exact_original_sha': query['raw_sha256'] == target['raw_sha256']}
            matches.append(item)
            by_query[query['dev_index']].append(item)
    def opposite_fit(i):
        return any(x['target_split'] == 'fit' and x['opposite_label'] for x in by_query[i])
    def exact_opposite_fit(i):
        return any(x['target_split'] == 'fit' and x['opposite_label'] and x['exact_original_sha'] for x in by_query[i])
    def any_fit(i):
        return any(x['target_split'] == 'fit' for x in by_query[i])
    def exact_fit(i):
        return any(x['target_split'] == 'fit' and x['exact_original_sha'] for x in by_query[i])
    def opposite_any(i):
        return any(x['opposite_label'] for x in by_query[i])
    def hard_dev_match(i):
        return any(x['target_split'] == 'dev' and x['target_dev_index'] in fixed for x in by_query[i])
    positive = {i for i, gi in enumerate(inputs['manifest']['dev_indices'])
                if inputs['manifest']['rows'][gi]['label'] == 1}
    remainder = positive - fixed
    reference = [dev[i]['quality'] if i in dev else None for i in sorted(remainder)]
    thresholds = quality_thresholds(reference)
    def distribution(indices):
        q = [dev[i]['quality'] for i in sorted(indices) if i in dev and dev[i]['quality'] is not None]
        return {'available': len(q), 'expected': len(indices), 'median': {
            k: float(np.median([x[k] for x in q])) if q else None
            for k in ('brightness', 'contrast', 'sharpness', 'field')}}
    ranked = {x['index']: x for x in inputs['ranked']}
    flags = {}
    for i in fixed:
        quality = quality_extreme(dev[i]['quality'] if i in dev else None, thresholds)
        consistent_uncertain = bool(ranked[i]['consensus'] == 'all' and ranked[i]['reviewed_uncertain'])
        flags[i] = {'opposite_label_candidate': opposite_any(i), 'quality_extreme': quality,
                    'consistent_below_majority_and_uncertain': consistent_uncertain}
    details = {'matches': matches, 'hard_positive_flags': flags}
    private = output / f'PRIVATE-MATCHES-{count:06d}.json'
    write(private, details)
    result = {'status': 'complete' if count == total else 'partial', 'processed': count, 'expected': total,
              'fit_processed': sum(x['split'] == 'fit' for x in rows),
              'dev_processed': len(dev), 'image_unknown': sum(x['hashes'] is None for x in rows),
              'fixed20_opposite_label_fit_candidates': sum(opposite_fit(i) for i in fixed),
              'fixed20_exact_opposite_label_fit_bytes': sum(exact_opposite_fit(i) for i in fixed),
              'all167_opposite_label_fit_candidates': sum(opposite_fit(i) for i in positive),
              'all167_any_fit_candidates': sum(any_fit(i) for i in positive),
              'all167_exact_fit_bytes': sum(exact_fit(i) for i in positive),
              'shared603_opposite_label_fit_candidates': sum(opposite_fit(i) for i in high),
              'shared241_opposite_label_fit_candidates': sum(opposite_fit(i) for i in high3),
              'shared603_exact_opposite_label_fit_bytes': sum(exact_opposite_fit(i) for i in high),
              'shared603_near_fixed20': sum(hard_dev_match(i) for i in high),
              'shared241_near_fixed20': sum(hard_dev_match(i) for i in high3),
              'quality_extreme_fixed20': sum(x['quality_extreme'] is True for x in flags.values()) if thresholds else None,
              'quality_reference_complete': thresholds is not None,
              'hard_data_risk_union': sum(any(v is True for v in row.values()) for row in flags.values()) if thresholds else None,
              'hard_without_risk_and_model_disagreement': sum(
                  not any(v is True for v in flags[i].values()) and ranked[i]['any_misorder_consensus'] == 'mixed'
                  for i in fixed) if thresholds else None,
              'quality': {'fixed20': distribution(fixed), 'remaining147': distribution(remainder),
                          'shared603': distribution(high), 'shared241': distribution(high3),
                          'dev_negative': distribution({i for i, gi in enumerate(inputs['manifest']['dev_indices'])
                                                        if inputs['manifest']['rows'][gi]['label'] == 0})},
              'private_matches_sha256': sha(private), 'official_read': False, 'GPU_initialized': False,
              'interpretation': 'Double perceptual hash is a visual candidate only; exact raw SHA proves equal bytes, not labeling error. Partial counts are lower bounds.'}
    summary = output / f'SUMMARY-{count:06d}.json'
    write(summary, result)
    print(json.dumps({**result, 'summary_sha256': sha(summary)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images-stage', choices=('init', 'scan', 'scan-adaptive', 'summary'))
    parser.add_argument('--site', type=Path)
    parser.add_argument('--site-sha')
    parser.add_argument('--original-source', type=Path)
    parser.add_argument('--max-images', type=int, default=200)
    args = parser.parse_args()
    if args.images_stage == 'init':
        image_init(args.base, args.source, args.output, args.site, args.site_sha)
    elif args.images_stage == 'scan':
        image_scan(args.base, args.source, args.output, args.site, args.site_sha, args.max_images)
    elif args.images_stage == 'scan-adaptive':
        if args.original_source is None:
            raise ValueError('the original sealed scanner source is required')
        image_scan_adaptive(args.base, args.source, args.original_source, args.output,
                            args.site, args.site_sha, args.max_images)
    elif args.images_stage == 'summary':
        image_summary(args.base, args.output, args.site, args.site_sha)
    else:
        ranks(args.base, args.source, args.packet, args.output)


if __name__ == '__main__':
    main()
