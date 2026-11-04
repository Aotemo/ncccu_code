"""Post-hoc opposite-label visual-similarity baseline for the sealed P3 scan.

Only aggregate output may leave exp-v3. Do not use this result for training,
relabeling, or ensemble selection. All image hashes and identities stay private.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from p3_hard_case import HashIndex, image_inputs, image_parts, sha, write


def count_opposite_matches(queries, targets):
    if (not queries or not targets or len({x['global_index'] for x in queries}) != len(queries)
            or len({x['global_index'] for x in targets}) != len(targets)
            or {x['global_index'] for x in queries} & {x['global_index'] for x in targets}
            or len({x['label'] for x in queries}) != 1 or len({x['label'] for x in targets}) != 1
            or queries[0]['label'] not in (0, 1) or targets[0]['label'] != 1 - queries[0]['label']):
        raise ValueError('disjoint complete opposite-label populations required')
    def convert(row):
        value = row.get('hashes')
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError('unknown image hash prevents complete comparison')
        try:
            pair = tuple(int(x, 16) for x in value)
        except (TypeError, ValueError) as error:
            raise ValueError('invalid image hash') from error
        if any(not 0 <= x < 2 ** 64 for x in pair):
            raise ValueError('invalid 64-bit image hash')
        return pair
    index = HashIndex([(i, convert(row)) for i, row in enumerate(targets)])
    matched = exact = pairs = 0
    for query in queries:
        found = index.find(convert(query))
        if found:
            matched += 1
            pairs += len(found)
            exact += any(query['raw_sha256'] == targets[i]['raw_sha256'] for i in found)
    return {'query_count': len(queries), 'target_count': len(targets),
            'matched_query_count': matched, 'matched_fraction': matched / len(queries),
            'candidate_pair_count': pairs, 'same_original_sha_query_count': exact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'scan-source', 'scan-output', 'site', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('site-sha', 'inputs-sha'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(k) != '' for k in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or not args.output.resolve().is_relative_to((args.base / 'P3-HARD-CASE').resolve())):
        raise ValueError('isolated hidden-device P3 output required')
    pinned = args.scan_output / 'INPUTS.json'
    if sha(pinned) != args.inputs_sha:
        raise ValueError('sealed scan inputs SHA differs')
    original_source = args.scan_source / 'tools/diagnostics/p3_hard_case.py'
    sealed = json.loads(pinned.read_text())
    if (sha(original_source) != sealed['source_sha256'] or sha(Path(__file__).parent / 'p3_hard_case.py') != sealed['source_sha256']
            or args.site_sha != sealed['site_sha256']):
        raise ValueError('published source or Pod identity differs from sealed scan')
    inputs = image_inputs(args.base, args.site, args.site_sha)
    if any(sealed[k] != v for k, v in inputs['identities'].items()):
        raise ValueError('sealed population identity differs')
    scanned = image_parts(args.scan_output, inputs['order'])
    dev = [row for row in scanned if row['split'] == 'dev']
    fit_positive = [row for row in scanned if row['split'] == 'fit' and row['label'] == 1]
    fit_negative = [row for row in scanned if row['split'] == 'fit' and row['label'] == 0]
    if len(dev) != 5140 or len(fit_positive) != 1490:
        raise ValueError('complete dev and fit-positive scan required before baseline')
    if any(row['hashes'] is None for row in dev + fit_positive):
        raise ValueError('unknown images prevent the complete negative baseline')
    shared = {x['dev_index'] for x in inputs['shared']['rows']}
    negative = [row for row in dev if row['label'] == 0]
    positive = [row for row in dev if row['label'] == 1]
    high = [row for row in negative if row['dev_index'] in shared]
    other = [row for row in negative if row['dev_index'] not in shared]
    if (len(negative), len(positive), len(high), len(other)) != (4973, 167, 603, 4370):
        raise ValueError('sealed baseline population counts differ')
    high_result = count_opposite_matches(high, fit_positive)
    other_result = count_opposite_matches(other, fit_positive)
    dev_pos_result = {'status': 'unknown', 'reason': 'fit negative scan incomplete',
                      'fit_negative_processed': len(fit_negative), 'fit_negative_expected': 44766}
    if len(fit_negative) == 44766 and all(row['hashes'] is not None for row in fit_negative):
        dev_pos_result = {'status': 'complete', **count_opposite_matches(positive, fit_negative)}
    elif len(fit_negative) == 44766:
        dev_pos_result = {'status': 'unknown', 'reason': 'fit negative image unknown',
                          'fit_negative_processed': len(fit_negative), 'fit_negative_expected': 44766,
                          'fit_negative_unknown': sum(row['hashes'] is None for row in fit_negative)}
    baseline_rate = other_result['matched_fraction']
    result = {'status': 'complete_dev_negative_baseline', 'at': time.time(),
              'scanned_count': len(scanned), 'expected_scan_count': 51396,
              'shared_603_to_fit_positive': high_result, 'other_4370_to_fit_positive': other_result,
              'rate_difference': high_result['matched_fraction'] - baseline_rate,
              'rate_ratio': high_result['matched_fraction'] / baseline_rate if baseline_rate else None,
              'dev_positive_to_fit_negative': dev_pos_result,
              'scan_inputs_sha256': args.inputs_sha, 'scan_source_sha256': sealed['source_sha256'],
              'source_sha256': sha(__file__), 'official_read': False, 'GPU_initialized': False,
              'interpretation': 'Post-hoc double-perceptual-hash candidates only; similarity does not establish patient identity, leakage, or a label error.'}
    write(args.output, result)
    print(json.dumps({**result, 'summary_sha256': sha(args.output)}), flush=True)


if __name__ == '__main__':
    main()
