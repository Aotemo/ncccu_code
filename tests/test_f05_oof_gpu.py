"""Synthetic fit-only role and supervised rank-packet checks; no real data."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'runtime/bundle'), str(ROOT / 'tools/diagnostics'), str(ROOT / 'tests')]
from retrain.age import make_spec
from retrain.persistence import canonical_digest
from retrain.parallel_evaluation import batch_assignments
from test_f05_oof import fixture
import f05_oof

GPU_MODULE = Path(os.environ.get('NCCCU_F04_MODULE', ROOT / 'tools/diagnostics/f04_gpu_full_dev.py'))
gpu_spec = importlib.util.spec_from_file_location('f04_gpu_full_dev', GPU_MODULE)
gpu = importlib.util.module_from_spec(gpu_spec)
gpu_spec.loader.exec_module(gpu)

MODULE = Path(os.environ.get('NCCCU_OOF_MODULE', ROOT / 'tools/diagnostics/f05_oof_gpu.py'))
if MODULE.is_file():
    spec = importlib.util.spec_from_file_location('f05_oof_gpu', MODULE)
    oof_gpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oof_gpu)
else:
    oof_gpu = None


def sealed_fixture():
    base = fixture()
    fold = f05_oof.build_fold_manifests(base, 20261121)[0]
    train = f05_oof.training_manifest(fold)
    sealed = {'fold': 0, 'round': 'R265', 'counts': fold['counts'],
              'fit_indices': fold['fit_indices'], 'audit_indices': fold['audit_indices'],
              'fit_indices_sha256': canonical_digest(fold['fit_indices']),
              'audit_indices_sha256': canonical_digest(fold['audit_indices']),
              'dev_indices_sha256': canonical_digest(fold['dev_indices']),
              'manifest_canonical_sha256': canonical_digest(fold), 'age': make_spec(fold)}
    return base, train, sealed


class FitOnlyOOFGPUTests(unittest.TestCase):
    def setUp(self):
        if oof_gpu is None:
            self.skipTest('OOF GPU helper has not yet been implemented')
        self.base, self.train, self.sealed = sealed_fixture()
        self.round_id = 'r265-small-f05fold0-seed20261029-pairsigmoid-24k-p4-9010-20261105'
        self.selected = {'variant': 'ema', 'tensor_sha256': 'a' * 64, 'successful_updates': 800}
        self.proofs = {str(rank): ('%064x' % (rank + 1)) for rank in range(8)}

    def packets(self, role):
        indices = self.sealed['fit_indices' if role == 'train' else 'audit_indices']
        result = []
        for rank in range(8):
            records = []
            for batch, batch_indices in batch_assignments(indices, 8, rank, 8):
                for offset, index in enumerate(batch_indices):
                    records.append({'index': index, 'rank': rank, 'batch': batch, 'offset': offset,
                                    'name': self.base['rows'][index]['name'],
                                    'label': self.base['rows'][index]['label'], 'logit': index / 100.,
                                    'probability': .5})
            result.append({'rank': rank, 'device_rank': rank, 'member': self.round_id,
                           'fold': 0, 'role': role, 'records': records, 'error': None,
                           'proof_sha256': self.proofs[str(rank)], 'input_sha256': 'b' * 64,
                           'model_load': {'selected_model_sha256': 'c' * 64,
                                          'selected_tensor_sha256': self.selected['tensor_sha256'],
                                          'variant': 'ema', 'successful_updates': 800,
                                          'strict_state_load': True, 'storage': 'fp32'},
                           'TTA': False, 'official_read': False})
        return result

    def score(self, role, packets):
        return oof_gpu.make_scores(self.base, self.sealed, packets, fold=0, role=role,
            round_id=self.round_id, inputs_sha='b' * 64, proof_shas=self.proofs,
            selected=self.selected, checkpoint_sha='c' * 64,
            base_manifest_sha='d' * 64, folds_sha='e' * 64)

    def test_plan_and_fold_role_partition(self):
        plan = oof_gpu.build_plan(self.round_id, 0)
        self.assertEqual([row['rank'] for row in plan], list(range(8)))
        self.assertTrue(all(row['member'] == self.round_id and row['world'] == 8 for row in plan))
        roles = oof_gpu.verified_roles(self.base, self.train, self.sealed, fold=0)
        self.assertEqual(roles['train'], self.sealed['fit_indices'])
        self.assertEqual(roles['heldout'], self.sealed['audit_indices'])
        for changed in (dict(self.sealed, fold=1),
                        dict(self.sealed, audit_indices=self.sealed['audit_indices'] + self.train['fit_indices'][:1]),
                        dict(self.sealed, audit_indices=self.sealed['audit_indices'] + self.base['dev_indices'][:1])):
            with self.assertRaises(ValueError):
                oof_gpu.verified_roles(self.base, self.train, changed, fold=0)

    def test_scores_cover_only_the_registered_role_and_preserve_order(self):
        for role, key in (('train', 'fit_indices'), ('heldout', 'audit_indices')):
            result = self.score(role, self.packets(role))
            self.assertEqual([row['index'] for row in result['rows']], self.sealed[key])
            self.assertEqual(result['role_indices_sha256'], canonical_digest(self.sealed[key]))
            self.assertTrue(all(set(row) == {'index', 'label', 'logit'} for row in result['rows']))
            self.assertEqual(result['selected_model_sha256'], 'c' * 64)

    def test_missing_duplicate_and_misaligned_packets_are_rejected(self):
        packets = self.packets('train')
        cases = [packets[:-1]]
        duplicate = copy.deepcopy(packets); duplicate[0]['records'].append(duplicate[0]['records'][0]); cases.append(duplicate)
        bad_proof = copy.deepcopy(packets); bad_proof[0]['proof_sha256'] = 'f' * 64; cases.append(bad_proof)
        bad_load = copy.deepcopy(packets); bad_load[0]['model_load']['selected_tensor_sha256'] = 'f' * 64; cases.append(bad_load)
        bad_role = copy.deepcopy(packets); bad_role[0]['role'] = 'heldout'; cases.append(bad_role)
        bad_official = copy.deepcopy(packets); bad_official[0]['official_read'] = True; cases.append(bad_official)
        bad_tta = copy.deepcopy(packets); bad_tta[0]['TTA'] = True; cases.append(bad_tta)
        bad_finite = copy.deepcopy(packets); bad_finite[0]['records'][0]['logit'] = float('nan'); cases.append(bad_finite)
        for candidate in cases:
            with self.subTest(candidate=cases.index(candidate)), self.assertRaises((ValueError, KeyError)):
                self.score('train', candidate)

    def test_evaluation_window_reserves_the_full_live_peer_budget(self):
        gib = 1024 ** 3
        peer_site = {'round_id': 'r266', 'round_root': '/work/rounds/r266', 'pod': {'uid': 'peer'}}
        peer = {'phase': 'running', 'formal_root': peer_site['round_root'], 'observed_at': 100.,
                'identity': {'owner_status': 'active', 'rank_status': ['active'] * 8},
                'signals_sent': False, 'model_forward_executed': False, 'official_read': False}
        window = {'status': 'f05_evaluation_capacity_verified', 'at': 100.,
                  'predecessor_safe_sha256': 'a' * 64, 'site_sha256': 'b' * 64,
                  'peer_observation_sha256': 'c' * 64, 'peer_site_sha256': 'd' * 64,
                  'peer_round_id': 'r266', 'peer_reserved_bytes': 12 * gib,
                  'peer_storage_limit_bytes': 12 * gib, 'required_free_bytes': 33 * gib,
                  'free_bytes': 40 * gib, 'training_activated': False,
                  'GPU_started': False, 'official_read': False}
        options = {'safe_sha': 'a' * 64, 'site_sha': 'b' * 64,
                   'peer_observation_sha': 'c' * 64, 'peer_site_sha': 'd' * 64,
                   'current_uid': 'current', 'now': 110., 'free': 40 * gib, 'peer_limit': 12 * gib}
        oof_gpu.validate_evaluation_window(window, peer, peer_site, **options)
        changes = [dict(window, peer_reserved_bytes=0), dict(window, at=-30.),
                   dict(window, official_read=True), dict(window, training_activated=True),
                   dict(window, required_free_bytes=32 * gib)]
        for changed in changes:
            with self.assertRaises(ValueError):
                oof_gpu.validate_evaluation_window(changed, peer, peer_site, **options)
        with self.assertRaises(ValueError):
            oof_gpu.validate_evaluation_window(window, peer, peer_site, **dict(options, free=33 * gib))
        with self.assertRaises(ValueError):
            oof_gpu.validate_evaluation_window(window, dict(peer, phase='unknown'), peer_site, **options)
        with self.assertRaises(ValueError):
            oof_gpu.validate_evaluation_window(window, peer, dict(peer_site, pod={'uid': 'current'}), **options)


class ImplementationExistsTest(unittest.TestCase):
    def test_oof_gpu_helper_exists(self):
        self.assertIsNotNone(oof_gpu, 'fit-only OOF GPU helper has not yet been implemented')


class OOFModeOptionsTest(unittest.TestCase):
    def test_fit_only_mode_never_accepts_soup_or_official_paths(self):
        gpu.validate_oof_options(True, False, None, None, None, None)
        gpu.validate_oof_options(False, True, None, None, None, None)
        for args in ((True, True, None, None, None, None),
                     (True, False, '/official', None, None, None),
                     (True, False, None, 'a' * 64, None, None),
                     (True, False, None, None, '/final', None),
                     (True, False, None, None, None, 'b' * 64)):
            with self.assertRaises(ValueError):
                gpu.validate_oof_options(*args)

    def test_standalone_evaluation_cannot_enter_official_or_untyped_mode(self):
        gpu.validate_standalone_options(True, True, False, None, None, None, None)
        gpu.validate_standalone_options(True, False, True, None, None, None, None)
        gpu.validate_standalone_options(False, False, False, None, None, None, None)
        for args in ((True, False, False, None, None, None, None),
                     (True, True, True, None, None, None, None),
                     (True, True, False, '/official', None, None, None),
                     (True, False, True, None, None, '/final', None)):
            with self.assertRaises(ValueError):
                gpu.validate_standalone_options(*args)


if __name__ == '__main__':
    print(json.dumps({'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'module_sha256': hashlib.sha256(MODULE.read_bytes()).hexdigest()
                      if MODULE.is_file() else None,
                      'gpu_source_sha256': hashlib.sha256(GPU_MODULE.read_bytes()).hexdigest()}), flush=True)
    unittest.main()
