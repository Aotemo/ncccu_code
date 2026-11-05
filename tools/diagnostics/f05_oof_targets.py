"""Fit-only F05 OOF calibration and target construction on the Pod CPU.

The producer is the separately supervised eight-rank inference chain. This
module never invokes a model, reads images, or consumes dev/official scores.
All sample-level outputs remain in a fresh private technical-prep directory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import warnings

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / 'runtime/bundle'), str(SOURCE / 'tools/diagnostics'),
                str(SOURCE / 'tools/pretraining')]

from retrain.classification_loss import f05_confident_joint
from retrain.data import F05_BASE_MANIFEST_SHA, F05_THREE_FOLD_SHA, validate_f05_target_table
from retrain.f05_folds import verify_bound_fold, verify_roles
from retrain.parallel_evaluation import canonical_predictions
from retrain.persistence import canonical_digest


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _path(reference, root):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'} or not _is_sha(reference['sha256']):
        raise ValueError('a pinned file path and SHA-256 are required')
    path, root = Path(reference['path']), Path(root).resolve(strict=True)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or not path.resolve(strict=True).is_relative_to(root) or _digest(path) != reference['sha256']):
        raise ValueError('pinned source file is missing, outside work root, or changed')
    return path


def _json(reference, root):
    try:
        value = json.loads(_path(reference, root).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('pinned JSON cannot be decoded') from error
    if not isinstance(value, dict):
        raise ValueError('pinned JSON object required')
    return value


def _source_files(freeze, frozen_root):
    paths = freeze.get('source_files')
    if not isinstance(paths, dict) or not paths:
        raise ValueError('selected historical runtime source identities required')
    bundle = frozen_root / 'bundle'
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError('historical runtime bundle missing')
    for name, sha in paths.items():
        if not isinstance(name, str):
            raise ValueError('invalid frozen runtime source reference')
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not _is_sha(sha):
            raise ValueError('invalid frozen runtime source reference')
        path = bundle / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(bundle.resolve()) or _digest(path) != sha:
            raise ValueError('historical runtime source bytes differ')


def _scores(reference, role, parent, sealed_fold, base, root):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'role_indices_sha256', 'count'}:
        raise ValueError('pinned score and exact population identity required')
    role_key = 'fit_indices' if role == 'train' else 'audit_indices'
    indices = sealed_fold[role_key]
    role_sha = sealed_fold[role_key + '_sha256']
    if (not isinstance(indices, list) or not indices or indices != sorted(indices)
            or len(indices) != len(set(indices)) or canonical_digest(indices) != role_sha
            or reference['role_indices_sha256'] != role_sha or reference['count'] != len(indices)):
        raise ValueError('sealed fold population identity differs')
    score = _json({'path': reference['path'], 'sha256': reference['sha256']}, root)
    expected = {'format': 'f05-fold-scores-v1', 'fold': parent['fold'], 'role': role,
                'base_manifest_sha256': parent['base_manifest_sha256'], 'folds_sha256': parent['folds_sha256'],
                'selected_model_sha256': parent['checkpoint']['sha256'],
                'selected_tensor_sha256': parent['selected']['tensor_sha256'],
                'selected_variant': parent['selected']['variant'], 'role_indices_sha256': role_sha}
    if set(score) != set(expected) | {'rows'} or any(score.get(k) != v for k, v in expected.items()):
        raise ValueError('score file and parent/fold/selected identity differ')
    rows = score['rows']
    if not isinstance(rows, list) or len(rows) != len(indices):
        raise ValueError('complete role scores required')
    for index, row in zip(indices, rows):
        if (not isinstance(row, dict) or set(row) != {'index', 'label', 'logit'}
                or type(row['index']) is not int or row['index'] != index
                or type(row['label']) is not int or row['label'] != base['rows'][index]['label']
                or type(row['logit']) not in (int, float) or not math.isfinite(row['logit'])):
            raise ValueError('score index, original label, or finite logit differs')
    return rows


def _packets(role, parent, indices, scores, base, release, proof_refs, root):
    packet_refs = parent[role + '_packets']
    if not isinstance(packet_refs, list) or len(packet_refs) != 8:
        raise ValueError('all eight role packets required')
    expected_load = {'selected_model_sha256': parent['checkpoint']['sha256'],
                     'selected_tensor_sha256': parent['selected']['tensor_sha256'],
                     'variant': parent['selected']['variant'],
                     'successful_updates': parent['selected']['successful_updates'],
                     'strict_state_load': True, 'storage': 'fp32'}
    packets = []
    for rank, reference in enumerate(packet_refs):
        packet = _json(reference, root)
        if (packet.get('rank') != rank or packet.get('device_rank') != rank
                or packet.get('member') != parent['round_id'] or packet.get('fold') != parent['fold']
                or packet.get('role') != role or packet.get('error') is not None
                or packet.get('proof_sha256') != proof_refs[rank]['sha256']
                or packet.get('input_sha256') != parent['producer_inputs']['sha256']
                or release['proof_sha256'].get(str(rank)) != proof_refs[rank]['sha256']
                or packet.get('model_load') != expected_load
                or packet.get('TTA') is not False or packet.get('official_read') is not False):
            raise ValueError('supervised rank packet or loaded selected identity differs')
        packets.append(packet)
    canonical = canonical_predictions(base['rows'], indices, packets, 8, 8)
    if len(canonical) != len(scores):
        raise ValueError('rank packet coverage differs')
    for produced, score in zip(canonical, scores):
        if produced['label'] != score['label'] or produced['logit'] != score['logit']:
            raise ValueError('private scores differ from verified rank packets')


def _verify_fold_input(base_ref, folds_ref, parent_ref, *, work_root, allow_fixture=False):
    """Recheck each byte, fold role, selected identity, and canonical rank output.

    ``allow_fixture`` exists only for synthetic tests; the command-line entry
    never enables it and also pins the actual base/fold SHA and population size.
    """
    base, sealed, parent = (_json(ref, work_root) for ref in (base_ref, folds_ref, parent_ref))
    if (parent.get('format') != 'f05-fold-inference-receipt-v1'
            or parent.get('status') != 'completed_fit_only_selected_logits'
            or parent.get('base_manifest_sha256') != base_ref['sha256']
            or parent.get('folds_sha256') != folds_ref['sha256']
            or sealed.get('status') != 'three_fold_fit_only_indices_sealed_no_training'
            or sealed.get('base_manifest_sha256') != base_ref['sha256'] or sealed.get('seed') != 20261121
            or sealed.get('training_started') is not False or sealed.get('official_read') is not False
            or type(parent.get('fold')) is not int or parent['fold'] not in (0, 1, 2)
            or parent.get('world_size') != 8 or parent.get('batch_size') != 8
            or parent.get('model_mode') != 'eval' or parent.get('view') != 'identity'
            or parent.get('TTA') is not False or parent.get('dev_read') is not False
            or parent.get('official_read') is not False):
        raise ValueError('completed fit-only fold inference identity required')
    fold_number = parent['fold']
    round_id = f'r{265 + fold_number}-small-f05fold{fold_number}-seed20261029-pairsigmoid-24k-p4-9010-20261105'
    if parent.get('round_id') != round_id or len(sealed.get('folds', [])) != 3:
        raise ValueError('sealed round or fold count differs')
    row = sealed['folds'][fold_number]
    train = _json(parent['training_manifest'], work_root)
    roles = verify_roles(base, train, row, fold_number=fold_number, split_seed=20261121)
    if parent.get('age_contract_sha256') != roles['age_contract_sha256']:
        raise ValueError('fold-only age identity differs')
    site = _json(parent['site'], work_root)
    science = _json(parent['science'], work_root)
    freeze = _json(parent['freeze'], work_root)
    completion = _json(parent['completion'], work_root)
    safe = _json(parent['safe'], work_root)
    sidecar = _json(parent['sidecar'], work_root)
    _path(parent['checkpoint'], work_root)
    selected = parent.get('selected')
    if (not isinstance(selected, dict) or set(selected) != {'variant', 'tensor_sha256', 'successful_updates'}
            or selected['variant'] not in ('online', 'ema') or not _is_sha(selected['tensor_sha256'])
            or type(selected['successful_updates']) is not int or selected['successful_updates'] <= 0
            or sidecar.get('sha256') != parent['checkpoint']['sha256']
            or freeze.get('blob_sha256') != parent['checkpoint']['sha256']
            or freeze.get('experiment91', {}).get('manifest_sha256') != parent['training_manifest']['sha256']
            or freeze.get('precision') != 'fp32' or freeze.get('view') != 'identity'
            or any(freeze.get('selected', {}).get(key) != value for key, value in selected.items())
            or completion.get('status') != 'completed' or type(completion.get('successful_updates')) is not int
            or completion['successful_updates'] < selected['successful_updates']
            or safe.get('phase') != 'terminal' or safe.get('done', {}).get('present') is not True
            or safe.get('done', {}).get('status') != 'completed' or safe.get('safety', {}).get('safe') is not True
            or safe.get('first_success_receipt_read') is not True or safe.get('signals_sent') is not False
            or safe.get('model_forward_executed') is not False or safe.get('official_read') is not False):
        raise ValueError('selected checkpoint, sidecar, completed run or safe identity differs')
    frozen_root = Path(parent['freeze']['path']).parent
    if (Path(parent['freeze']['path']).name != 'SINGLE-MODEL-FREEZE.json'
            or Path(parent['completion']['path']).name != 'TRAIN_RESULT.json'
            or Path(parent['completion']['path']).parent not in (frozen_root, frozen_root / 'evidence')
            or safe.get('formal_root') != str(frozen_root)):
        raise ValueError('selected freeze, completion and safety are not the same round')
    tree = _json(parent['source_tree_proof'], work_root)
    if allow_fixture:
        if not (freeze.get('cpu_fixture_only') is True and site.get('fixture_only') is True
                and science.get('fixture_only') is True and tree.get('fixture_only') is True):
            raise ValueError('fixture mode requires explicit synthetic sources')
    else:
        if (freeze.get('cpu_fixture_only') is not False or frozen_root.name != round_id
                or site.get('round_id') != round_id
                or Path(parent['site']['path']) != frozen_root / 'SITE_BINDING.json'
                or tree.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
                or tree.get('GPU_initialized') is not False or tree.get('official_read') is not False):
            raise ValueError('real published source and original fold run are required')
        _source_files(freeze, frozen_root)
        bound = verify_bound_fold(site, science)
        if any(bound.get(key) != roles[key] for key in ('fit_indices_sha256', 'heldout_indices_sha256', 'age_contract_sha256')):
            raise ValueError('original site and sealed fold binding differ')
        if Path(parent['training_manifest']['path']) != Path(site['metadata_root']) / 'manifest.json':
            raise ValueError('training manifest is not the site-bound fold file')
    inputs = _json(parent['producer_inputs'], work_root)
    spec = _json(parent['worker_spec'], work_root)
    release = _json(parent['release'], work_root)
    if (inputs.get('fold') != fold_number or inputs.get('base_manifest_sha256') != base_ref['sha256']
            or inputs.get('folds_sha256') != folds_ref['sha256']
            or inputs.get('training_manifest_sha256') != parent['training_manifest']['sha256']
            or inputs.get('checkpoint_sha256') != parent['checkpoint']['sha256']
            or inputs.get('selected') != selected or spec.get('inputs') != parent['producer_inputs']
            or release.get('status') != 'all_rank_device_proofs_passed'
            or release.get('spec_sha256') != parent['worker_spec']['sha256']
            or not isinstance(release.get('proof_sha256'), dict)):
        raise ValueError('producer inputs, worker specification or device release differs')
    if not allow_fixture:
        source_root = Path(tree.get('source_root', ''))
        worker = source_root / 'tools/diagnostics/f04_gpu_full_dev.py'
        if (not source_root.is_absolute() or spec.get('source_root') != str(source_root)
                or worker.is_symlink() or not worker.is_file()
                or _digest(worker) != spec.get('worker_source_sha256')):
            raise ValueError('published producer source differs')
    proof_refs = parent.get('rank_proofs')
    if not isinstance(proof_refs, list) or len(proof_refs) != 8:
        raise ValueError('eight device proofs required')
    for rank, reference in enumerate(proof_refs):
        proof = _json(reference, work_root)
        if proof.get('rank') != rank or release['proof_sha256'].get(str(rank)) != reference['sha256']:
            raise ValueError('rank proof does not match supervised release')
    if not isinstance(parent.get('scores'), dict) or set(parent['scores']) != {'train', 'heldout'}:
        raise ValueError('both fit-only score populations required')
    scores = {role: _scores(parent['scores'][role], role, parent, row, base, work_root)
              for role in ('train', 'heldout')}
    counts = {role: len(scores[role]) for role in scores}
    if parent.get('forward_counts') != counts:
        raise ValueError('one forward per declared role row required')
    for role, key in (('train', 'fit_indices'), ('heldout', 'audit_indices')):
        _packets(role, parent, row[key], scores[role], base, release, proof_refs, work_root)
    return {'fold': fold_number, 'round_id': round_id, 'selected_model_sha256': parent['checkpoint']['sha256'],
            'selected_tensor_sha256': selected['tensor_sha256'], 'selected_variant': selected['variant'],
            'selected_successful_updates': selected['successful_updates'], 'base_manifest_sha256': base_ref['sha256'],
            'folds_sha256': folds_ref['sha256'], 'parent_sha256': parent_ref['sha256'],
            'train_role_sha256': row['fit_indices_sha256'], 'heldout_role_sha256': row['audit_indices_sha256'],
            'train_score_sha256': parent['scores']['train']['sha256'],
            'heldout_score_sha256': parent['scores']['heldout']['sha256'],
            'train_rows': scores['train'], 'heldout_rows': scores['heldout']}


def verify_fold_input(base_ref, folds_ref, parent_ref, *, work_root, allow_fixture=False):
    try:
        return _verify_fold_input(base_ref, folds_ref, parent_ref, work_root=work_root, allow_fixture=allow_fixture)
    except (KeyError, TypeError, IndexError, OSError) as error:
        raise ValueError('complete pinned fold evidence is required') from error


def fit_fold_calibration(verified):
    """Fit on every fold-training row with unit sample weight, then map heldout."""
    import numpy as np
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    train, heldout = verified['train_rows'], verified['heldout_rows']
    labels = [row['label'] for row in train]
    if not train or not heldout or set(labels) != {0, 1}:
        raise ValueError('complete fold training calibration needs both classes')
    x = np.asarray([row['logit'] for row in train], dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(x).all():
        raise ValueError('finite training logits required')
    weights = np.ones(len(train), dtype=np.float64)
    model = LogisticRegression(C=1000, solver='lbfgs', max_iter=1000, tol=1e-8, random_state=20261121)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    if (any(issubclass(w.category, ConvergenceWarning) for w in caught)
            or list(model.classes_) != [0, 1] or len(model.n_iter_) != 1 or int(model.n_iter_[0]) >= 1000
            or model.coef_.shape != (1, 1) or model.intercept_.shape != (1,)
            or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all()):
        raise ValueError('fold calibration did not converge to finite binary parameters')
    hold = np.asarray([row['logit'] for row in heldout], dtype=np.float64).reshape(-1, 1)
    probabilities = model.predict_proba(hold)[:, 1]
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('finite heldout probabilities in [0,1] required')
    receipt = {'format': 'f05-fold-calibration-v1', 'fold': verified['fold'], 'round_id': verified['round_id'],
               'base_manifest_sha256': verified['base_manifest_sha256'], 'folds_sha256': verified['folds_sha256'],
               'parent_sha256': verified['parent_sha256'], 'selected_model_sha256': verified['selected_model_sha256'],
               'selected_tensor_sha256': verified['selected_tensor_sha256'],
               'selected_variant': verified['selected_variant'],
               'selected_successful_updates': verified['selected_successful_updates'],
               'train_role_sha256': verified['train_role_sha256'], 'train_score_sha256': verified['train_score_sha256'],
               'training_count': len(train), 'class_counts': [labels.count(0), labels.count(1)],
               'weight_per_sample': 1.0, 'weight_sum': float(weights.sum()), 'classes': [0, 1],
               'config': {'C': 1000, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 1e-8, 'random_state': 20261121},
               'coef': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0]),
               'iterations': int(model.n_iter_[0]), 'converged': True, 'sklearn_version': sklearn.__version__}
    return receipt, {row['index']: float(prob) for row, prob in zip(heldout, probabilities)}


def build_target_tables(base, sealed, results, *, base_sha, folds_sha):
    """Create clean and soft-label tables from three already verified folds."""
    from scipy.special import expit
    if len(results) != 3 or [result['verified']['fold'] for result in results] != [0, 1, 2]:
        raise ValueError('three ordered complete fold results required')
    by_index, membership, sources = {}, {}, []
    for result in results:
        verified, calibration, probabilities = result['verified'], result['calibration'], result['probabilities']
        number = verified['fold']
        encoded = json.dumps(calibration, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                             allow_nan=False).encode('utf-8') + b'\n'
        if (verified['base_manifest_sha256'] != base_sha or verified['folds_sha256'] != folds_sha
                or not _is_sha(result['calibration_sha256'])
                or hashlib.sha256(encoded).hexdigest() != result['calibration_sha256']
                or calibration.get('format') != 'f05-fold-calibration-v1' or calibration.get('converged') is not True
                or calibration.get('fold') != number or calibration.get('round_id') != verified['round_id']
                or calibration.get('base_manifest_sha256') != base_sha or calibration.get('folds_sha256') != folds_sha
                or calibration.get('selected_model_sha256') != verified['selected_model_sha256']
                or calibration.get('selected_tensor_sha256') != verified['selected_tensor_sha256']
                or calibration.get('selected_variant') != verified['selected_variant']
                or calibration.get('selected_successful_updates') != verified['selected_successful_updates']
                or calibration.get('train_score_sha256') != verified['train_score_sha256']
                or calibration.get('train_role_sha256') != verified['train_role_sha256']
                or calibration.get('parent_sha256') != verified['parent_sha256']
                or calibration.get('training_count') != len(verified['train_rows'])
                or calibration.get('weight_per_sample') != 1.0
                or calibration.get('weight_sum') != float(len(verified['train_rows']))
                or calibration.get('classes') != [0, 1]
                or calibration.get('config') != {'C': 1000, 'solver': 'lbfgs', 'max_iter': 1000,
                                                  'tol': 1e-8, 'random_state': 20261121}
                or type(calibration.get('coef')) not in (int, float)
                or type(calibration.get('intercept')) not in (int, float)
                or not math.isfinite(calibration['coef']) or not math.isfinite(calibration['intercept'])
                or set(probabilities) != set(sealed['folds'][number]['audit_indices'])):
            raise ValueError('fold-specific calibrated heldout lineage or coverage differs')
        for held in verified['heldout_rows']:
            expected = float(expit(calibration['coef'] * held['logit'] + calibration['intercept']))
            if not math.isclose(probabilities[held['index']], expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('OOF probability differs from the fixed fold calibration mapping')
        for index, value in probabilities.items():
            if (index in by_index or type(index) is not int or type(value) not in (int, float)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('duplicate, invalid or nonfinite OOF probability')
            by_index[index] = float(value); membership[index] = number
        sources.append({'fold': number, 'selected_model_sha256': verified['selected_model_sha256'],
                        'calibration_sha256': result['calibration_sha256']})
    indices = sorted(base['fit_indices'])
    if len(indices) != len(set(indices)) or set(by_index) != set(indices) or set(indices) & set(base['dev_indices']):
        raise ValueError('OOF must cover original fit exactly once and exclude dev')
    joint = f05_confident_joint([base['rows'][i]['label'] for i in indices], [by_index[i] for i in indices])
    suspicious = set(joint['suspicious_positions'])
    common = {'format': 'f05-fit-oof-targets-v1', 'base_manifest_sha256': base_sha,
              'folds_sha256': folds_sha, 'fold_sources': sources}
    clean_rows, soft_rows = [], []
    for position, index in enumerate(indices):
        identity = {'index': index, 'label': base['rows'][index]['label'], 'fold': membership[index]}
        clean_rows.append({**identity, 'weight': .25 if position in suspicious else 1.})
        soft_rows.append({**identity, 'soft_target': by_index[index]})
    clean = {**common, 'mode': 'clean', 'rows': clean_rows}
    distilled = {**common, 'mode': 'distill', 'rows': soft_rows}
    summary = {'fit_count': len(indices), 'thresholds': joint['thresholds'], 'confident_joint': joint['confident_joint'],
               'suspicious_count': len(suspicious), 'suspicious_fraction': len(suspicious) / len(indices),
               'unknown_count': len(joint['unknown_positions']), 'official_read': False, 'dev_read': False}
    return clean, distilled, summary


def _write_new(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', nargs=2, metavar=('PATH', 'SHA256'), required=True)
    parser.add_argument('--folds', nargs=2, metavar=('PATH', 'SHA256'), required=True)
    parser.add_argument('--parent', nargs=2, action='append', metavar=('PATH', 'SHA256'), required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from source_chain import cpu_context, configure_torch_threads
    context = cpu_context(); configure_torch_threads(context)
    base_ref = dict(zip(('path', 'sha256'), args.base))
    folds_ref = dict(zip(('path', 'sha256'), args.folds))
    parent_refs = [dict(zip(('path', 'sha256'), pair)) for pair in args.parent]
    if (base_ref['sha256'] != F05_BASE_MANIFEST_SHA or folds_ref['sha256'] != F05_THREE_FOLD_SHA
            or len(parent_refs) != 3):
        raise ValueError('registered original manifest, three-fold seal and three parents required')
    base, sealed = _json(base_ref, args.work_root), _json(folds_ref, args.work_root)
    if len(base['fit_indices']) != 46256 or len(base['dev_indices']) != 5140:
        raise ValueError('registered 9:1 original population required')
    output, work = args.output, args.work_root.resolve(strict=True)
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.resolve(strict=True).is_relative_to(work / 'technical-prep')):
        raise ValueError('new independent technical-prep output required')
    verified = [verify_fold_input(base_ref, folds_ref, ref, work_root=work) for ref in parent_refs]
    if [item['fold'] for item in verified] != [0, 1, 2]:
        raise ValueError('parents must be supplied in registered fold order')
    output.mkdir()
    results, calibration_refs = [], []
    for item in verified:
        calibration, probabilities = fit_fold_calibration(item)
        reference = _write_new(output / ('CALIBRATION-' + str(item['fold']) + '.json'), calibration)
        calibration_refs.append(reference)
        results.append({'verified': item, 'calibration': calibration,
                        'calibration_sha256': reference['sha256'], 'probabilities': probabilities})
    clean, distilled, summary = build_target_tables(base, sealed, results,
        base_sha=base_ref['sha256'], folds_sha=folds_ref['sha256'])
    probabilities = {index: value for result in results for index, value in result['probabilities'].items()}
    sources = clean['fold_sources']
    for table in (clean, distilled):
        validate_f05_target_table(base, sealed, table, sources, probabilities,
                                  base_sha=base_ref['sha256'], folds_sha=folds_ref['sha256'])
    clean_ref = _write_new(output / 'CLEAN-TARGETS-private.json', clean)
    distill_ref = _write_new(output / 'DISTILL-TARGETS-private.json', distilled)
    suspects = [row['index'] for row in clean['rows'] if row['weight'] == .25]
    suspect_ref = _write_new(output / 'SUSPICIOUS-private.json', {'indices': suspects, 'mode': 'confidence_disagreement_not_diagnosis'})
    summary.update(status='completed_fit_only_oof_targets', inputs={'base': base_ref, 'folds': folds_ref,
        'parents': parent_refs}, calibrations=calibration_refs,
        heldout_score_sha256=[item['heldout_score_sha256'] for item in verified],
        clean_targets=clean_ref, distill_targets=distill_ref, suspicious=suspect_ref,
        execution=context, limitations=['Calibration uses fold training scores and may be overconfident.',
        'Internal dev selected the fold models; OOF is not an independent generalization estimate.'])
    report_ref = _write_new(output / 'SUMMARY.json', summary)
    print(json.dumps({'status': summary['status'], 'fit_count': summary['fit_count'],
                      'suspicious_count': summary['suspicious_count'], 'summary_sha256': report_ref['sha256'],
                      'clean_sha256': clean_ref['sha256'], 'distill_sha256': distill_ref['sha256']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
