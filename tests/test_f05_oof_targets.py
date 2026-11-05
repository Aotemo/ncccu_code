"""Synthetic-only fold score/receipt fixtures and F05 target tests.

Every row, checkpoint, rank packet and process proof below is fabricated in a
temporary directory. Nothing in this fixture demonstrates a real GPU run.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/bundle'), str(ROOT / 'tools/diagnostics'), str(ROOT / 'tests')]

from retrain.age import make_spec
from retrain.persistence import canonical_digest
from test_f05_oof import fixture as base_fixture
import f05_oof

SOURCE = ROOT / 'tools/diagnostics/f05_oof_targets.py'
if SOURCE.is_file():
    spec = importlib.util.spec_from_file_location('f05_oof_targets', SOURCE)
    targets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(targets)
else:
    targets = None


def _ref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def _write(root, name, obj):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
    path.write_bytes(raw)
    return _ref(path)


def synthetic_fixture(root):
    """Emit complete fabricated three-fold sources using the documented format."""
    root = Path(root).resolve()
    base = base_fixture()
    base_ref = _write(root, 'base.json', base)
    fold_manifests = f05_oof.build_fold_manifests(base, 20261121)
    sealed_rows = []
    for number, fold in enumerate(fold_manifests):
        sealed_rows.append({'fold': number, 'round': 'R' + str(265 + number), 'counts': fold['counts'],
                            'fit_indices': fold['fit_indices'], 'audit_indices': fold['audit_indices'],
                            'fit_indices_sha256': canonical_digest(fold['fit_indices']),
                            'audit_indices_sha256': canonical_digest(fold['audit_indices']),
                            'dev_indices_sha256': canonical_digest(fold['dev_indices']),
                            'manifest_canonical_sha256': canonical_digest(fold), 'age': make_spec(fold)})
    sealed = {'status': 'three_fold_fit_only_indices_sealed_no_training', 'seed': 20261121,
              'base_manifest_sha256': base_ref['sha256'], 'folds': sealed_rows,
              'GPU_initialized': False, 'training_started': False, 'official_read': False}
    folds_ref = _write(root, 'folds.json', sealed)
    parents = []
    for number, fold in enumerate(fold_manifests):
        directory = root / ('fold-' + str(number))
        directory.mkdir()
        round_id = f'r{265 + number}-small-f05fold{number}-seed20261029-pairsigmoid-24k-p4-9010-20261105'
        train = f05_oof.training_manifest(fold)
        train_ref = _write(directory, 'training-manifest.json', train)
        checkpoint = directory / 'fixture-checkpoint.pt'
        checkpoint.write_bytes(('synthetic checkpoint ' + str(number)).encode('utf-8'))
        checkpoint_ref = _ref(checkpoint)
        variant = 'online' if number == 1 else 'ema'
        selected = {'variant': variant, 'tensor_sha256': str(number + 1) * 64,
                    'successful_updates': 800}
        sidecar = _write(directory, 'fixture-checkpoint.pt.json', {'sha256': checkpoint_ref['sha256']})
        freeze = _write(directory, 'SINGLE-MODEL-FREEZE.json', {
            'cpu_fixture_only': True, 'precision': 'fp32', 'view': 'identity',
            'selected': selected, 'blob_sha256': checkpoint_ref['sha256'],
            'experiment91': {'manifest_sha256': train_ref['sha256']}, 'source_files': {}})
        completion = _write(directory, 'TRAIN_RESULT.json', {'status': 'completed', 'successful_updates': 800})
        safe = _write(directory, 'SAFE.json', {'fixture_only': True, 'phase': 'terminal',
            'formal_root': str(directory), 'done': {'present': True, 'status': 'completed'},
            'safety': {'safe': True}, 'first_success_receipt_read': True, 'signals_sent': False,
            'model_forward_executed': False, 'official_read': False})
        site = _write(directory, 'SITE_BINDING.json', {'fixture_only': True, 'round_id': round_id})
        science = _write(directory, 'SCIENCE.json', {'fixture_only': True, 'round_id': round_id})
        tree = _write(directory, 'LOCAL-TREE.json', {'fixture_only': True})
        inputs = _write(directory, 'INPUTS.json', {'fixture_only': True, 'fold': number,
            'base_manifest_sha256': base_ref['sha256'], 'folds_sha256': folds_ref['sha256'],
            'training_manifest_sha256': train_ref['sha256'], 'checkpoint_sha256': checkpoint_ref['sha256'],
            'selected': selected})
        worker_spec = _write(directory, 'WORKER-SPEC.json', {'fixture_only': True, 'inputs': inputs})
        proofs = [_write(directory, f'proof-{rank}.json', {'fixture_only': True, 'rank': rank})
                  for rank in range(8)]
        release = _write(directory, 'RELEASE.json', {'status': 'all_rank_device_proofs_passed',
            'fixture_only': True, 'spec_sha256': worker_spec['sha256'],
            'proof_sha256': {str(rank): proof['sha256'] for rank, proof in enumerate(proofs)}})
        scores, packets = {}, {}
        for role, indices in (('train', fold['fit_indices']), ('heldout', fold['audit_indices'])):
            role_digest = canonical_digest(indices)
            rows = [{'index': i, 'label': base['rows'][i]['label'],
                     'logit': (i - 50) / 25 + (0.4 if base['rows'][i]['label'] else -0.4)}
                    for i in indices]
            body = {'format': 'f05-fold-scores-v1', 'fold': number, 'role': role,
                    'base_manifest_sha256': base_ref['sha256'], 'folds_sha256': folds_ref['sha256'],
                    'selected_model_sha256': checkpoint_ref['sha256'],
                    'selected_tensor_sha256': selected['tensor_sha256'],
                    'selected_variant': variant, 'role_indices_sha256': role_digest, 'rows': rows}
            scores[role] = {**_write(directory, role + '-scores.json', body),
                            'role_indices_sha256': role_digest, 'count': len(rows)}
            rank_refs = []
            for rank in range(8):
                records = []
                for ordinal, start in enumerate(range(0, len(rows), 8)):
                    if ordinal % 8 != rank:
                        continue
                    for offset, row in enumerate(rows[start:start + 8]):
                        index, logit = row['index'], row['logit']
                        records.append({**row, 'rank': rank, 'batch': ordinal, 'offset': offset,
                                        'name': base['rows'][index]['name'],
                                        'probability': 1 / (1 + math.exp(-logit))})
                packet = {'rank': rank, 'device_rank': rank, 'member': round_id, 'fold': number,
                          'role': role, 'error': None, 'records': records,
                          'proof_sha256': proofs[rank]['sha256'], 'input_sha256': inputs['sha256'],
                          'model_load': {'selected_model_sha256': checkpoint_ref['sha256'],
                                         'selected_tensor_sha256': selected['tensor_sha256'],
                                         'variant': variant, 'successful_updates': 800,
                                         'strict_state_load': True, 'storage': 'fp32'},
                          'TTA': False, 'official_read': False}
                rank_refs.append(_write(directory, f'{role}-rank-{rank}.json', packet))
            packets[role] = rank_refs
        receipt = {'format': 'f05-fold-inference-receipt-v1', 'status': 'completed_fit_only_selected_logits',
                   'fold': number, 'round_id': round_id, 'base_manifest_sha256': base_ref['sha256'],
                   'folds_sha256': folds_ref['sha256'], 'training_manifest': train_ref, 'site': site,
                   'science': science, 'freeze': freeze, 'completion': completion, 'safe': safe,
                   'checkpoint': checkpoint_ref, 'sidecar': sidecar,
                   'age_contract_sha256': sealed_rows[number]['age']['contract_sha256'],
                   'selected': selected, 'producer_inputs': inputs, 'worker_spec': worker_spec,
                   'release': release, 'source_tree_proof': tree, 'rank_proofs': proofs,
                   'train_packets': packets['train'], 'heldout_packets': packets['heldout'],
                   'scores': scores, 'forward_counts': {role: value['count'] for role, value in scores.items()},
                   'world_size': 8, 'batch_size': 8, 'model_mode': 'eval', 'view': 'identity',
                   'TTA': False, 'dev_read': False, 'official_read': False}
        parents.append(_write(directory, 'FOLD-INFERENCE.json', receipt))
    return {'base': base_ref, 'folds': folds_ref, 'parents': parents}


class OOFTargetTests(unittest.TestCase):
    def test_target_module_exists(self):
        self.assertIsNotNone(targets, 'F05 target consumer has not been implemented')

    def test_synthetic_fixture_has_complete_pinned_parent_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = synthetic_fixture(directory)
            self.assertEqual(len(fixture['parents']), 3)
            for number, reference in enumerate(fixture['parents']):
                receipt = json.loads(Path(reference['path']).read_text())
                self.assertEqual(receipt['fold'], number)
                self.assertEqual(receipt['world_size'], 8)
                for role in ('train', 'heldout'):
                    score = receipt['scores'][role]
                    actual = _ref(Path(score['path']))
                    self.assertEqual(score['sha256'], actual['sha256'])
                    rows = json.loads(Path(score['path']).read_text())['rows']
                    self.assertEqual(score['role_indices_sha256'], canonical_digest([r['index'] for r in rows]))
                    self.assertEqual(receipt['forward_counts'][role], len(rows))

    @unittest.skipIf(targets is None, 'consumer not implemented')
    def test_full_train_calibration_and_fit_only_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = synthetic_fixture(directory)
            base = json.loads(Path(fixture['base']['path']).read_text())
            sealed = json.loads(Path(fixture['folds']['path']).read_text())
            verified = [targets.verify_fold_input(fixture['base'], fixture['folds'], parent,
                                                 work_root=directory, allow_fixture=True)
                        for parent in fixture['parents']]
            results = []
            for item in verified:
                calibration, probabilities = targets.fit_fold_calibration(item)
                expected_count = len(sealed['folds'][item['fold']]['fit_indices'])
                self.assertEqual(calibration['training_count'], expected_count)
                self.assertEqual(calibration['weight_per_sample'], 1.0)
                self.assertEqual(calibration['weight_sum'], float(expected_count))
                self.assertTrue(calibration['converged'])
                self.assertEqual(calibration['classes'], [0, 1])
                self.assertEqual(calibration['selected_model_sha256'], item['selected_model_sha256'])
                reference = _write(Path(directory), f'calibration-{item["fold"]}.json', calibration)
                results.append({'verified': item, 'calibration': calibration,
                                'calibration_sha256': reference['sha256'], 'probabilities': probabilities})
            clean, distilled, summary = targets.build_target_tables(base, sealed, results,
                base_sha=fixture['base']['sha256'], folds_sha=fixture['folds']['sha256'])
            self.assertEqual([r['index'] for r in clean['rows']], base['fit_indices'])
            self.assertEqual([r['index'] for r in distilled['rows']], base['fit_indices'])
            self.assertTrue(all(r['weight'] in (.25, 1.) for r in clean['rows']))
            self.assertTrue(all(0 <= r['soft_target'] <= 1 for r in distilled['rows']))
            self.assertEqual(summary['fit_count'], len(base['fit_indices']))
            self.assertEqual([s['fold'] for s in clean['fold_sources']], [0, 1, 2])
            self.assertEqual([s['selected_model_sha256'] for s in clean['fold_sources']],
                             [i['selected_model_sha256'] for i in verified])
            self.assertEqual(verified[1]['selected_variant'], 'online')
            self.assertEqual(verified[0]['selected_variant'], 'ema')
            broken = copy.deepcopy(results)
            broken[0]['probabilities'].pop(next(iter(broken[0]['probabilities'])))
            with self.assertRaises(ValueError):
                targets.build_target_tables(base, sealed, broken,
                    base_sha=fixture['base']['sha256'], folds_sha=fixture['folds']['sha256'])
            broken = copy.deepcopy(results)
            first = next(iter(broken[0]['probabilities']))
            broken[0]['probabilities'][first] = 1.0 - broken[0]['probabilities'][first]
            with self.assertRaises(ValueError):
                targets.build_target_tables(base, sealed, broken,
                    base_sha=fixture['base']['sha256'], folds_sha=fixture['folds']['sha256'])
            broken = copy.deepcopy(results)
            broken[0]['calibration']['coef'] += 1
            with self.assertRaises(ValueError):
                targets.build_target_tables(base, sealed, broken,
                    base_sha=fixture['base']['sha256'], folds_sha=fixture['folds']['sha256'])

    @unittest.skipIf(targets is None, 'consumer not implemented')
    def test_rejects_stale_score_wrong_variant_and_missing_train_row(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = synthetic_fixture(directory)
            first = fixture['parents'][0]
            parent = json.loads(Path(first['path']).read_text())
            wrong = copy.deepcopy(parent)
            wrong['selected']['variant'] = 'online'
            forged = _write(Path(directory), 'wrong-variant.json', wrong)
            with self.assertRaises(ValueError):
                targets.verify_fold_input(fixture['base'], fixture['folds'], forged,
                                          work_root=directory, allow_fixture=True)
            score_ref = parent['scores']['train']
            score_path = Path(score_ref['path'])
            score = json.loads(score_path.read_text())
            score['rows'].pop()
            _write(score_path.parent, score_path.name, score)
            with self.assertRaises(ValueError):
                targets.verify_fold_input(fixture['base'], fixture['folds'], first,
                                          work_root=directory, allow_fixture=True)


if __name__ == '__main__':
    files = ('tools/diagnostics/f05_oof_targets.py', 'tests/test_f05_oof_targets.py',
             'runtime/bundle/retrain/classification_loss.py', 'runtime/bundle/retrain/data.py',
             'runtime/bundle/retrain/f05_folds.py', 'runtime/bundle/retrain/parallel_evaluation.py')
    print(json.dumps({'source_sha256': {name: _ref(ROOT / name)['sha256'] for name in files}}), flush=True)
    unittest.main()
