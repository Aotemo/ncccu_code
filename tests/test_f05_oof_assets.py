"""Fold assets retain dev identity and exclude held-out rows from training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'runtime/bundle'), str(root / 'tools/diagnostics'), str(root / 'tests')]
import f05_oof
from test_f05_oof import fixture
from retrain.persistence import canonical_digest

spec = importlib.util.spec_from_file_location('f05_oof_assets', root / 'tools/diagnostics/f05_oof_assets.py')
assets = importlib.util.module_from_spec(spec); spec.loader.exec_module(assets)


class FoldAssetsTests(unittest.TestCase):
    def test_metadata_derivation_rejects_stale_or_five_fold_policy(self):
        policy = {'base_manifest_sha256':'a' * 64, 'sealed_folds_sha256':'b' * 64,
                  'folds':3, 'fold_seed':20261121, 'official_read':False}
        self.assertTrue(assets.validate_registered_policy(policy, 'a' * 64, 'b' * 64))
        for key, wrong in (('folds', 5), ('sealed_folds_sha256', 'c' * 64),
                           ('fold_seed', 20261122), ('official_read', True)):
            changed = dict(policy, **{key:wrong})
            with self.subTest(key=key), self.assertRaises(ValueError):
                assets.validate_registered_policy(changed, 'a' * 64, 'b' * 64)

    def test_fold_files_bind_train_dev_age_original_and_independent_seal(self):
        base = fixture(); fold = f05_oof.build_fold_manifests(base, 20261121)[0]
        schedule = b'{"steps_per_cycle":800}\n'
        files = assets.construct_assets(fold, {'source_identity': base['source_identity']}, schedule,
                                        base_manifest_sha256='a' * 64, sealed_folds_sha256='b' * 64,
                                        base_freeze_sha256='c' * 64)
        manifest, age, contract, freeze = [json.loads(files[name]) for name in
                                            ('manifest.json', 'AGE.json', 'CONTRACT.json', 'FREEZE.json')]
        self.assertEqual(manifest['fit_indices'], fold['fit_indices'])
        self.assertEqual(manifest['dev_indices'], base['dev_indices'])
        self.assertEqual(manifest['audit_indices'], [])
        self.assertEqual(age['contract']['fit_indices_sha256'], canonical_digest(fold['fit_indices']))
        self.assertEqual(contract['new_manifest']['sha256'], hashlib.sha256(files['manifest.json']).hexdigest())
        self.assertEqual(contract['f05_fold']['heldout_indices_sha256'], canonical_digest(fold['audit_indices']))
        self.assertEqual(contract['f05_fold']['sealed_folds_sha256'], 'b' * 64)
        self.assertEqual(contract['f05_fold']['folds'], 3)
        for name, entry in freeze['files'].items():
            self.assertEqual(entry['sha256'], hashlib.sha256(files[name]).hexdigest())

    def test_fold_files_reject_wrong_source_or_overlap(self):
        base = fixture(); fold = f05_oof.build_fold_manifests(base, 20261121)[0]
        with self.assertRaises(ValueError):
            assets.construct_assets(fold, {'source_identity': {'other': True}}, b'{}',
                                    base_manifest_sha256='a' * 64, sealed_folds_sha256='b' * 64,
                                    base_freeze_sha256='c' * 64)
        fold['split']['seed'] = 20261122
        with self.assertRaises(ValueError):
            assets.construct_assets(fold, {'source_identity': base['source_identity']}, b'{"steps_per_cycle":800}',
                                    base_manifest_sha256='a' * 64, sealed_folds_sha256='b' * 64,
                                    base_freeze_sha256='c' * 64)
        fold['split']['seed'] = 20261121
        fold['audit_indices'].append(fold['fit_indices'][0])
        with self.assertRaises(ValueError):
            assets.construct_assets(fold, {'source_identity': base['source_identity']}, b'{}',
                                    base_manifest_sha256='a' * 64, sealed_folds_sha256='b' * 64,
                                    base_freeze_sha256='c' * 64)


if __name__ == '__main__':
    unittest.main()
