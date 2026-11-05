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
                          'TTA': False, 'official_read': False}
                rank_refs.append(_write(directory, f'{role}-rank-{rank}.json', packet))
            packets[role] = rank_refs
        receipt = {'format': 'f05-fold-inference-receipt-v1', 'status': 'completed_fit_only_selected_logits',
                   'fold': number, 'round_id': round_id, 'base_manifest_sha256': base_ref['sha256'],
                   'folds_sha256': folds_ref['sha256'], 'training_manifest': train_ref, 'site': site,
                   'science': science, 'freeze': freeze, 'completion': completion,
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


if __name__ == '__main__':
    unittest.main()
