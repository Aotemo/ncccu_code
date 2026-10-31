"""One bounded, fit-only joint-input label consistency check for frozen R120.

Only sealed manifest metadata is read. No model, image, cache, or prediction is
opened. This is an aggregate of (sealed image SHA, actual FP32 age features).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROUND = 'r120-small22k-seed20261029-scale085-pairsigmoid-24k-p4-9010-20261031'
COMMIT = '7922dd788f982bcb72d30a1a8e6d6e7761e42e88'
FREEZE_SHA = '5f808304a0e4ae84501525a1634f0ca55ec80e16bcba67342a66e7e211f5bcf2'
SITE_SHA = '26cae5aefe899351d153dff9c1dd6d685c129dcc0cb578bc73a2f03af409ce43'
MAX_SECONDS = 300
FIT_COUNT = 46256


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    if not path.is_file() or path.is_symlink():
        raise ValueError('required metadata file missing or symlinked')
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    started = time.monotonic()
    os.umask(0o077)
    if len(sys.argv) != 4:
        raise SystemExit('usage: r120_fit_input_consistency.py ROUND_ROOT R107_FREEZE_JSON OUTPUT_DIR')
    root, r107_path, output = map(Path, sys.argv[1:])
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / 'START.json').write_text(json.dumps({
        'schema': 'r120-fit-input-consistency-start-v1', 'scope': 'fit_metadata_only',
        'max_seconds': MAX_SECONDS, 'passes': 1, 'model_or_image_cache_bytes': False},
        sort_keys=True) + '\n', encoding='utf-8')
    result = {'schema': 'r120-fit-input-consistency-v1', 'status': 'unknown',
              'scope': 'same_sealed_image_sha_and_fp32_age_features_fit_only',
              'round': ROUND, 'source_git_commit': COMMIT, 'no_model_or_image_cache_bytes': True,
              'no_fit256_retry': True, 'reuse_or_compute': 'computed_from_existing_sealed_fit_metadata'}
    stage = 'frozen_metadata_identity'
    try:
        require(root.name == ROUND, 'round root differs')
        freeze_path, site_path, assets_path = (root / name for name in
                                                ('SINGLE-MODEL-FREEZE.json', 'SITE_BINDING.json', 'ASSETS.json'))
        require(digest(freeze_path) == FREEZE_SHA and digest(site_path) == SITE_SHA,
                'original R120 frozen metadata SHA differs')
        freeze, site, assets, r107 = map(read_json,
            (freeze_path, site_path, assets_path, r107_path))
        require(site['round_id'] == ROUND and site['round_root'] == str(root)
                and site['git_commit'] == COMMIT
                and site['pod']['uid'] == os.environ['NCCCU_POD_UID'],
                'R120 site identity differs')
        sealed = assets['manifests']['fit_dev']
        manifest_path = Path(sealed['remote_path'])
        require(manifest_path == Path(site['metadata_root']) / 'manifest.json'
                and digest(manifest_path) == sealed['sha256'],
                'sealed manifest path or byte SHA differs')
        manifest = read_json(manifest_path)
        sys.path.insert(0, str(root / 'bundle'))
        import numpy as np
        from retrain.age import features, validate_spec
        from retrain.persistence import canonical_digest
        require('torch' not in sys.modules, 'model framework imported unexpectedly')
        exp = freeze['experiment91']
        old = r107['experiment91']
        fields = ('manifest_sha256', 'manifest_digest', 'fit_indices_sha256',
                  'age_contract_sha256')
        require(all(exp[field] == old[field] for field in fields),
                'R107/R120 sealed fit or age contract identity differs')
        require(exp['manifest_sha256'] == sealed['sha256']
                and exp['manifest_digest'] == canonical_digest(manifest)
                and exp['fit_indices_sha256'] == canonical_digest(manifest['fit_indices']),
                'R120 manifest identity differs')
        spec = freeze['job']['age']
        validate_spec(spec)
        require(spec['enabled'] is True
                and spec['contract_sha256'] == exp['age_contract_sha256']
                and spec['contract']['fit_indices_sha256'] == exp['fit_indices_sha256']
                and freeze['job']['model'] == 'convnext_small',
                'R120 original age or job binding differs')
        result.update({'identity_passed': True,
                       'manifest_sha256': exp['manifest_sha256'],
                       'manifest_digest': exp['manifest_digest'],
                       'fit_indices_sha256': exp['fit_indices_sha256'],
                       'age_contract_sha256': exp['age_contract_sha256'],
                       'r107_freeze_metadata_sha256': digest(r107_path),
                       'r107_r120_fit_age_contract_equal': True})
        stage = 'single_fit_metadata_pass'
        rows, indices = manifest['rows'], manifest['fit_indices']
        require(type(indices) is list and len(indices) == FIT_COUNT
                and len(set(indices)) == FIT_COUNT,
                'fit indices cardinality or uniqueness differs')
        counts = {}
        positive = negative = 0
        for index in indices:
            require(type(index) is int and 0 <= index < len(rows), 'fit index invalid')
            row = rows[index]
            label, sealed_sha = row.get('label'), row.get('sha256')
            require(type(label) is int and label in (0, 1)
                    and type(sealed_sha) is str and len(sealed_sha) == 64
                    and all(char in '0123456789abcdef' for char in sealed_sha),
                    'fit label or sealed image SHA invalid')
            pair = features(row, spec)
            require(len(pair) == 2, 'age feature dimensionality differs')
            age_value, missing_value = (float(np.float32(value)) for value in pair)
            require(math.isfinite(age_value) and math.isfinite(missing_value),
                    'nonfinite FP32 age feature')
            age_value = 0.0 if age_value == 0.0 else age_value
            missing_value = 0.0 if missing_value == 0.0 else missing_value
            key = (sealed_sha, age_value, missing_value)
            values = counts.setdefault(key, [0, 0])
            values[label] += 1
            positive += label
            negative += 1 - label
        require(positive + negative == FIT_COUNT
                and sum(sum(values) for values in counts.values()) == FIT_COUNT,
                'fit aggregate count differs')
        duplicate = [values for values in counts.values() if sum(values) >= 2]
        cross = [values for values in counts.values() if values[0] and values[1]]
        cross_negative = sum(values[0] for values in cross)
        cross_positive = sum(values[1] for values in cross)
        conflict_pairs = sum(values[0] * values[1] for values in cross)
        require(len(cross) <= len(duplicate) and conflict_pairs >= len(cross)
                and cross_positive <= positive and cross_negative <= negative,
                'joint-key aggregation invariants differ')
        result.update({'status': 'complete', 'fit_count': FIT_COUNT,
                       'positive_count': positive, 'negative_count': negative,
                       'duplicate_joint_key_count': len(duplicate),
                       'duplicate_joint_key_row_count': sum(sum(v) for v in duplicate),
                       'cross_label_joint_key_count': len(cross),
                       'cross_label_positive_row_count': cross_positive,
                       'cross_label_negative_row_count': cross_negative,
                       'cross_label_pair_count_C': conflict_pairs,
                       'finite_all_fp32_age_features': True})
    except BaseException as error:
        result['status'] = 'unknown'
        result['failure_stage'] = stage
        result['failure_type'] = type(error).__name__
        (output / 'ERROR-PRIVATE.txt').write_text(repr(error) + '\n', encoding='utf-8')
    result['elapsed_seconds'] = time.monotonic() - started
    if result['elapsed_seconds'] > MAX_SECONDS:
        result = {key: value for key, value in result.items()
                  if key not in ('fit_count', 'positive_count', 'negative_count',
                                 'duplicate_joint_key_count', 'duplicate_joint_key_row_count',
                                 'cross_label_joint_key_count', 'cross_label_positive_row_count',
                                 'cross_label_negative_row_count', 'cross_label_pair_count_C')}
        result['status'] = 'unknown'
        result['failure_stage'] = 'time_budget'
    result['within_300_seconds'] = result['elapsed_seconds'] <= MAX_SECONDS
    result['pod_uid'] = os.environ.get('NCCCU_POD_UID')
    result['cpu_mask'] = sorted(os.sched_getaffinity(0))
    result_path = output / 'RESULT.json'
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    print(json.dumps({'result_path': str(result_path),
                      'result_sha256': digest(result_path), 'result': result},
                     sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
