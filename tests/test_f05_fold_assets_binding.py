"""Derived three-fold assets must remain bound to original and sealed data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'runtime/bundle'), str(root / 'tools/diagnostics'), str(root / 'tests')]
import f05_oof
import f05_oof_assets
from retrain import age
from retrain import f05_folds
from retrain.persistence import canonical_digest
from test_f05_oof import fixture


class FoldAssetsBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root_dir = Path(self.temporary.name)
        self.root_dir = root_dir
        self.base_root, self.fold_root = root_dir / 'base', root_dir / 'technical-prep/fold'
        self.base_root.mkdir(); self.fold_root.mkdir(parents=True)
        base = fixture(); folds = f05_oof.build_fold_manifests(base, 20261121)
        raw = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
        base_files = {'manifest.json':raw(base), 'CONTRACT.json':raw({'source_identity':base['source_identity']}),
                      'AGE.json':raw(age.make_spec(base)), 'SCHEDULE.json':raw({'steps_per_cycle':800})}
        for name, value in base_files.items():
            (self.base_root / name).write_bytes(value)
        freeze = {'status':'SEALED', 'files':{name:{'sha256':hashlib.sha256(value).hexdigest()}
                                             for name,value in base_files.items()}}
        (self.base_root / 'FREEZE.json').write_bytes(raw(freeze))
        sealed_rows = []
        for number, fold in enumerate(folds):
            sealed_rows.append({'fold':number, 'round':'R' + str(265 + number), 'counts':fold['counts'],
                'fit_indices':fold['fit_indices'], 'audit_indices':fold['audit_indices'],
                'fit_indices_sha256':canonical_digest(fold['fit_indices']),
                'audit_indices_sha256':canonical_digest(fold['audit_indices']),
                'dev_indices_sha256':canonical_digest(fold['dev_indices']),
                'manifest_canonical_sha256':canonical_digest(fold), 'age':age.make_spec(fold)})
        seal = {'status':'three_fold_fit_only_indices_sealed_no_training', 'seed':20261121,
                'base_manifest_sha256':hashlib.sha256(base_files['manifest.json']).hexdigest(),
                'folds':sealed_rows, 'GPU_initialized':False, 'training_started':False, 'official_read':False}
        self.sealed_path = root_dir / 'technical-prep/FOLDS.json'; self.sealed_path.write_bytes(raw(seal))
        self.sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.base_sha = self.sha(self.base_root / 'FREEZE.json')
        self.manifest_sha = self.sha(self.base_root / 'manifest.json')
        self.sealed_sha = self.sha(self.sealed_path)
        files = f05_oof_assets.construct_assets(folds[0], {'source_identity':base['source_identity']},
                  base_files['SCHEDULE.json'], base_manifest_sha256=self.manifest_sha,
                  sealed_folds_sha256=self.sealed_sha, base_freeze_sha256=self.base_sha)
        for name, value in files.items():
            (self.fold_root / name).write_bytes(value)
        self.fold_sha = self.sha(self.fold_root / 'FREEZE.json')

    def verify(self):
        self.assertTrue(hasattr(f05_folds, 'verify_assets'), 'bound fold verifier is required')
        return f05_folds.verify_assets(self.fold_root, self.base_root, self.sealed_path, fold_number=0,
                                       base_freeze_sha256=self.base_sha, base_manifest_sha256=self.manifest_sha,
                                       sealed_folds_sha256=self.sealed_sha, derived_freeze_sha256=self.fold_sha)

    def test_original_sealed_and_derived_assets_pass(self):
        result = self.verify()
        self.assertEqual(result['fold_number'], 0)
        self.assertEqual(result['derived_freeze_sha256'], self.fold_sha)

    def test_changed_derived_or_original_metadata_is_rejected(self):
        path = self.fold_root / 'AGE.json'; original = path.read_bytes(); path.write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            self.verify()
        path.write_bytes(original)
        path = self.base_root / 'SCHEDULE.json'; path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            self.verify()

    def test_changed_sealed_fold_is_rejected(self):
        self.sealed_path.write_bytes(self.sealed_path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            self.verify()

    def test_resealed_fold_with_changed_dev_roles_is_rejected(self):
        # Even a self-consistent derived freeze cannot override the original
        # fit/dev identity and the independently sealed held-out indices.
        path = self.fold_root / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['dev_indices'][0] = manifest['fit_indices'][0]
        raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode()
        path.write_bytes(raw)
        freeze_path = self.fold_root / 'FREEZE.json'
        freeze = json.loads(freeze_path.read_text())
        freeze['files']['manifest.json']['sha256'] = hashlib.sha256(raw).hexdigest()
        freeze['files']['manifest.json']['bytes'] = len(raw)
        freeze_path.write_text(json.dumps(freeze, sort_keys=True, separators=(',', ':')) + '\n')
        self.fold_sha = self.sha(freeze_path)
        with self.assertRaises(ValueError):
            self.verify()

    def test_site_and_science_must_bind_the_original_migration_and_fold(self):
        self.assertTrue(hasattr(f05_folds, 'verify_bound_fold'), 'site-bound fold verifier is required')
        profile = ('single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-'
                   'seed20261029-scale085-pairsigmoid-f05fold0-v1')
        site = {'work_root':str(self.root_dir), 'metadata_root':str(self.fold_root),
                'round_id':'r265-small-f05fold0-seed20261029-pairsigmoid-24k-p4-9010-20261105'}
        science = {'candidate_profile':profile, 'paths':{'metadata_root':'technical-prep/fold'},
                   'f05_fold':{'version':'fit-only-three-fold-v1', 'fold':0, 'folds':3,
                               'split_seed':20261121, 'base_metadata_root':'base',
                               'base_freeze_sha256':self.base_sha, 'base_manifest_sha256':self.manifest_sha,
                               'sealed_folds_path':'technical-prep/FOLDS.json',
                               'sealed_folds_sha256':self.sealed_sha, 'derived_freeze_sha256':self.fold_sha}}
        with patch.object(f05_folds, 'ORIGINAL_FREEZE_SHA', self.base_sha), \
             patch.object(f05_folds, 'ORIGINAL_MANIFEST_SHA', self.manifest_sha), \
             patch.object(f05_folds, 'THREE_FOLD_SEAL_SHA', self.sealed_sha):
            result = f05_folds.verify_bound_fold(site, science, str(self.base_root))
            self.assertEqual(result['fold_number'], 0)
            with self.assertRaises(ValueError):
                f05_folds.verify_bound_fold(site, science, str(self.fold_root))
            wrong = dict(site, round_id=site['round_id'].replace('r265', 'r266'))
            with self.assertRaises(ValueError):
                f05_folds.verify_bound_fold(wrong, science, str(self.base_root))
            wrong_science = dict(science, paths={'metadata_root':'technical-prep/other'})
            with self.assertRaises(ValueError):
                f05_folds.verify_bound_fold(site, wrong_science, str(self.base_root))
            spec = importlib.util.spec_from_file_location('fold_prepare', root / 'tools/activation/prepare.py')
            prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)
            self.assertTrue(hasattr(prepare, 'verify_fold_migration'), 'formal preparation must bind migration')
            self.assertTrue(prepare.verify_fold_migration(site, science,
                            {'paths':{'stage1_root':str(self.base_root)}}))
            with self.assertRaises(ValueError):
                prepare.verify_fold_migration(site, science, {'paths':{'stage1_root':str(self.fold_root)}})
            spec = importlib.util.spec_from_file_location('fold_gpu_site', root / 'tools/activation/prepare_gpu_site.py')
            gpu_site = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpu_site)
            self.assertTrue(hasattr(gpu_site, 'verify_fold_relocation'), 'GPU site must bind relocation')
            self.assertTrue(gpu_site.verify_fold_relocation(site, science,
                            {'paths':{'stage1_root':str(self.base_root)}}))
            with self.assertRaises(ValueError):
                gpu_site.verify_fold_relocation(site, science, {'paths':{'stage1_root':str(self.fold_root)}})
            self.assertFalse(gpu_site.verify_fold_relocation(site,
                             {'candidate_profile':'original-non-fold'}, {'paths':{'stage1_root':str(self.base_root)}}))
            from retrain import experiment91 as ex
            site['science'] = {'configuration':{'sha256':'f' * 64}}
            science['schedule'] = {'steps_per_cycle':800, 'total_updates':24000,
                                   'warmup_updates':1600, 'stop_after_updates':24000}
            with patch.object(ex, 'load_site', return_value=site), \
                 patch.object(ex, 'science_configuration', return_value=science), \
                 patch.object(ex, 'candidate', return_value={'model':'convnext_small'}):
                ex._metadata.cache_clear()
                self.addCleanup(ex._metadata.cache_clear)
                self.assertEqual(ex._metadata(str(self.fold_root))['binding']['freeze_sha256'], self.fold_sha)
                original = self.base_root / 'manifest.json'
                original.write_bytes(original.read_bytes() + b' ')
                ex._metadata.cache_clear()
                with self.assertRaises(ValueError):
                    ex._metadata(str(self.fold_root))


if __name__ == '__main__':
    unittest.main()
