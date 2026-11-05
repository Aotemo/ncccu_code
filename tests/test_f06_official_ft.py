"""F06 official-fold manifest, scientific binding and warm-start identity."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import age as age_api, config, data, fresh  # noqa: E402
from retrain.experiment91 import F06_BINDING  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OfficialFoldManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = [{'name': f'train_{i:03d}.jpg', 'label': i % 2, 'age': '60',
                 'group': f'g{i:03d}', 'relative_path': f'train/train_{i:03d}.jpg'} for i in range(40)]
        valid = [{'name': f'valid_{i:03d}.jpg', 'label': 1 if i in (0, 1, 4, 5) else 0, 'age': '70',
                  'group': f'sha{i:03d}', 'relative_path': f'valid/valid_{i:03d}.JPG'}
                 for i in range(12)]
        self.source = {'rows': rows, 'valid_rows': valid,
                       'identity': {'train_count': 40, 'valid_count': 12}}
        entries, fold_rows = [], [0] * 12
        for fold, width in enumerate((4, 4, 4)):
            fold_rows[fold * 4:(fold + 1) * 4] = [fold] * width
        for row, fold in zip(valid, fold_rows):
            entries.append({'name': row['name'], 'label': row['label'], 'fold': fold})
        self.folds = self.root / 'FOLDS.json'
        self.folds.write_text(json.dumps({
            'format': 'f06-official-folds-v1', 'population': {'rows': 12, 'positive': 4},
            'folds': [{'fold': 0, 'rows': 4, 'positive': 2}, {'fold': 1, 'rows': 4, 'positive': 2},
                      {'fold': 2, 'rows': 4, 'positive': 0}],
            'rows': entries}, sort_keys=True) + '\n')
        self.split = {'kind': 'official_cv', 'seed': 20261024, 'valid_fraction': .1,
                      'folds': 3, 'fold': 0, 'inner_fraction': .1, 'include_valid': True,
                      'folds_path': str(self.folds), 'folds_sha256': _sha(self.folds)}

    def test_fold_roles_use_unseen_official_rows_only(self):
        manifest = data.build_manifest(self.source, self.split)
        self.assertEqual(len(manifest['rows']), 52)
        self.assertEqual(len(manifest['audit_indices']), 0)
        self.assertTrue(all(i >= 40 for i in manifest['dev_indices']))
        self.assertEqual({manifest['rows'][i]['name'] for i in manifest['dev_indices']},
                         {f'valid_{i:03d}.jpg' for i in range(4)})
        fit_names = {manifest['rows'][i]['name'] for i in manifest['fit_indices']}
        self.assertFalse(fit_names & {f'valid_{i:03d}.jpg' for i in range(4)})
        self.assertIn('valid_004.jpg', fit_names)
        self.assertEqual(manifest['counts']['dev']['positive'], 2)

    def test_fold_registration_is_mandatory_and_exact(self):
        bad = dict(self.split, folds_sha256='0' * 64)
        with self.assertRaises(ValueError):
            data.build_manifest(self.source, bad)
        with self.assertRaises(ValueError):
            data.build_manifest(self.source, dict(self.split, include_valid=False))
        with self.assertRaises(ValueError):
            data.build_manifest(self.source, dict(self.split, expected_fold_counts=[[4, 1], [4, 2], [4, 0]]))
        self.split['expected_fold_counts'] = [[4, 2], [4, 2], [4, 0]]
        manifest = data.build_manifest(self.source, self.split)
        self.assertEqual(len(manifest['dev_indices']), 4)


class F06ScienceBindingTest(unittest.TestCase):
    def setUp(self):
        job = config.default_job('f06', 'convnext_small', 'checkpoint_warm_start')
        job.update(image_size=512, prepared_size=512, precision='fp32', normalization='imagenet', cycles=15, warmup_cycles=1,
                   min_cycles=1, patience=4, min_delta=0.0, stop_after_cycles=0, backbone_lr=1e-5,
                   head_lr=1e-4, weight_decay=1e-4, dropout=.2, stochastic_depth=.1,
                   head_lr_scope='last_linear', physical_batch=32, effective_batch=64,
                   positive_per_batch=8, final_norm_eps=1e-6,
                   loss={'version': 'binary-pairwise-sigmoid-v1', 'kind': 'binary_pairwise_sigmoid',
                         'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32',
                         'scope': 'global_batch'},
                   training_scale=[.85, 1.0], seed=F06_BINDING['seed'] + 1,
                   f06={'version': F06_BINDING['version'], 'fold': 1,
                        'folds_sha256': F06_BINDING['folds_sha256']},
                   age=age_api.make_spec({'rows': [{'name': 'a.jpg', 'sha256': 'a' * 64, 'age': '60'},
                                                  {'name': 'b.jpg', 'sha256': 'b' * 64, 'age': '50'}],
                                          'fit_indices': [0, 1], 'source_identity': {}}),
                   initialization={'kind': 'checkpoint_warm_start', 'format': 'exported-fp32-state-dict-v1',
                                    'path': '/tmp/warm.pt', 'source_model': '/tmp/checkpoint.pt',
                                    'sha256': 'b' * 64,
                                    'tensor_sha256': F06_BINDING['warm_start_tensor_sha256']})
        self.job = job

    def test_registered_science_and_fold_seed(self):
        config.validate_job(self.job)
        changed = dict(self.job, backbone_lr=1e-4)
        with self.assertRaises(ValueError):
            config.validate_job(changed)
        changed = dict(self.job, f06=dict(self.job['f06'], folds_sha256='0' * 64))
        with self.assertRaises(ValueError):
            config.validate_job(changed)

    def test_split_kind_requires_fold_registration(self):
        split = {'kind': 'official_cv', 'seed': 20261024, 'valid_fraction': .1, 'folds': 3, 'fold': 0,
                 'inner_fraction': .1, 'include_valid': True, 'folds_path': '/tmp/FOLDS.json',
                 'folds_sha256': 'a' * 64, 'expected_fold_counts': [[1950, 64], [1950, 64], [1949, 63]]}
        config.validate_split(split)
        with self.assertRaises(ValueError):
            config.validate_split(dict(split, include_valid=False))
        with self.assertRaises(ValueError):
            config.validate_split(dict(split, kind='holdout'))

    def test_registered_step_clock(self):
        self.assertEqual(fresh.resolve_steps_per_cycle(self.job, 48000, fresh_spec={'x': 1}), 400)


class WarmStartIdentityTest(unittest.TestCase):
    def test_strict_tensor_identity_round_trip(self):
        import torch
        from retrain import models
        from retrain.ema import tensor_digest
        torch.manual_seed(7)
        job = {'model': 'convnext_small', 'dropout': .2, 'stochastic_depth': .1,
               'final_norm_eps': 1e-6, 'initialization': {'kind': 'random'}, 'age': None}
        model = models.create_model(job, initialize=False)
        state = {name: tensor.detach().clone().float() for name, tensor in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'warm.pt'
            torch.save(state, path)
            init = {'kind': 'checkpoint_warm_start', 'format': 'exported-fp32-state-dict-v1',
                    'path': str(path), 'source_model': '/tmp/checkpoint.pt',
                    'sha256': _sha(path), 'tensor_sha256': tensor_digest(state)}
            job['initialization'] = init
            warm = models.create_model(job, initialize=True)
            self.assertEqual(tensor_digest(warm.state_dict()), init['tensor_sha256'])
            tampered = dict(state)
            name = sorted(tampered)[0]
            tampered[name] = tampered[name] + 1
            torch.save(tampered, path)
            with self.assertRaises(ValueError):
                models.create_model(dict(job, initialization=dict(init, sha256=_sha(path))), initialize=True)


if __name__ == '__main__':
    unittest.main()
