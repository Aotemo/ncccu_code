"""Independent reference-stream oracle for the fixed fit-label class quota."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


class FitLabelQuotaTest(unittest.TestCase):
    def test_eight_rank_quota_reference_order_seeds_and_iterator_independence(self):
        from retrain import ssl_stage as ssl
        from retrain.data import _seed
        self.assertTrue(callable(getattr(ssl, 'FitClassQuotaSampler', None)),
            'fixed fit class-quota sampler missing')
        class FitRow:
            def __init__(self, label): self.label = label
            def __getitem__(self, key):
                if key != 'label': raise AssertionError('sampler read age or other row data')
                return self.label
        class DevRow:
            def __getitem__(self, key): raise AssertionError('sampler read a dev row')
        manifest = {'rows': [FitRow(int(i < 13)) for i in range(160)] + [DevRow()] * 4,
            'fit_indices': list(range(160)), 'dev_indices': list(range(160, 164)), 'audit_indices': []}
        seed, updates = 563721, 70
        samplers = [ssl.FitClassQuotaSampler(manifest, seed=seed, updates=updates, rank=rank, world=8)
            for rank in range(8)]
        streams = [list(sampler) for sampler in samplers]
        self.assertTrue(all(len(sampler) == len(stream) == updates
            for sampler, stream in zip(samplers, streams)))
        pools = {1: list(range(13)), 0: list(range(13, 160))}
        cache = {}
        def expected_index(label, offset):
            epoch, within = divmod(offset, len(pools[label]))
            key = label, epoch
            if key not in cache:
                values = pools[label][:]
                random.Random(_seed(seed, 'permutation', label, epoch)).shuffle(values)
                cache[key] = values
            return cache[key][within]
        positive_draws = 0
        for update in range(updates):
            actual = [item for rank in range(8) for item in streams[rank][update]]
            expected_indices = []
            for micro in range(2):
                expected_indices += [expected_index(1, update*8 + micro*4 + k) for k in range(4)]
                expected_indices += [expected_index(0, update*56 + micro*28 + k) for k in range(28)]
            expected = [(index, _seed(seed, 'sample', update, position, index))
                for position, index in enumerate(expected_indices)]
            self.assertEqual(actual, expected)
            self.assertEqual(sum(index < 13 for index, _ in actual), 8)
            self.assertEqual(len(actual), 64)
            self.assertTrue(all(index in manifest['fit_indices'] for index, _ in actual))
            positive_draws += sum(index < 13 for index, _ in actual)
        self.assertEqual(positive_draws, updates*8)
        self.assertEqual(list(samplers[0]), streams[0])
        partial = iter(samplers[0])
        for _ in range(19): next(partial)
        self.assertEqual(list(samplers[0]), streams[0])
        self.assertEqual(list(partial), streams[0][19:])
        for kwargs in ({'world': 4}, {'rank': 8}, {'updates': 0}):
            values = dict(seed=seed, updates=updates, rank=0, world=8); values.update(kwargs)
            with self.assertRaises(ValueError): ssl.FitClassQuotaSampler(manifest, **values)
        overlap = dict(manifest, dev_indices=[0, 160])
        with self.assertRaises(ValueError):
            ssl.FitClassQuotaSampler(overlap, seed=seed, updates=updates, rank=0, world=8)
        one_class = dict(manifest, rows=[FitRow(0) for _ in range(160)] + [DevRow()] * 4)
        with self.assertRaises(ValueError):
            ssl.FitClassQuotaSampler(one_class, seed=seed, updates=updates, rank=0, world=8)

    def test_fixed_recipe_completion_and_original_supervision(self):
        from retrain import ssl_stage as ssl, experiment91 as ex
        self.assertTrue(hasattr(ssl, 'FIT_LABEL_QUOTA_RECIPE'), 'fixed class-quota recipe missing')
        expected = dict(ssl.FIT_LABEL_RECIPE, version='fit-label-supcon-quota8-1600-v1',
            sampling_policy='fit-class-quota-g64-p8-v1', global_positive_quota=8)
        self.assertEqual(ssl.FIT_LABEL_QUOTA_RECIPE, expected)
        complete = dict(status='completed', recipe=expected, successful_updates=1600,
            attempted_updates=1600, skipped_updates=0, dev_evaluations=0,
            labels_or_age_used_for_loss=True, labels_used_for_loss=True,
            age_used_for_loss=False, label_scope='fit_csv',
            fit_sampling_policy='fit-class-quota-g64-p8-v1', fit_global_positive_quota=8,
            fit_positive_primary_draws=12800, fit_negative_primary_draws=89600,
            contrastive_fit_positive_images=8)
        self.assertTrue(ssl.ssl_phase_completed(complete, expected))
        for changed in ({'fit_sampling_policy': 'uniform'}, {'fit_global_positive_quota': 7},
                {'fit_positive_primary_draws': 12799}, {'fit_negative_primary_draws': 89599},
                {'contrastive_fit_positive_images': 7}, {'age_used_for_loss': True}, {'label_scope': 'dev_csv'}):
            self.assertFalse(ssl.ssl_phase_completed(dict(complete, **changed), expected))
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        indices = {'fit_indices': list(range(128)), 'dev_indices': [128], 'audit_indices': []}
        for label, prior_label, seed, order, binding in (
            ('R240', 'R238', 20261029, 155, ex.FIT_LABEL_QUOTA_SMALL_BINDING),
            ('R241', 'R239', 20261027, 156, ex.FIT_LABEL_QUOTA_SMALL_SEED27_BINDING)):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            job = copy.deepcopy(original['job']); job['seed'] = seed
            self.assertEqual(value['job'], job)
            self.assertEqual(value['candidate_profile'], binding)
            self.assertEqual(value['candidate']['experiment_order'], order)
            self.assertEqual(value['ssl_recipe'], expected)
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            for key in ('public_source', 'paths', 'data_split', 'schedule', 'early_stopping', 'structure'):
                self.assertEqual(value[key], original[key])
            prior = json.loads((ROOT / 'configs' / (prior_label + '-SCIENCE.json')).read_text())
            current_rng = ssl.ssl_rng_derivation(expected, seed, 8)
            prior_rng = ssl.ssl_rng_derivation(prior['ssl_recipe'], seed, 8)
            for key in ('learning_domain', 'sampling_domain', 'projector_seed', 'model_seeds', 'sampler_seed', 'loader_seeds'):
                self.assertEqual(current_rng[key], prior_rng[key])
            self.assertEqual(ssl.validate_recipe(expected, job, indices, 8), expected)
            with self.assertRaises(ValueError): ssl.validate_recipe(dict(expected, global_positive_quota=7), job, indices, 8)
        for label in ('R115', 'R228', 'R230', 'R232', 'R234', 'R236', 'R238', 'R239'):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            ssl.validate_recipe(value['ssl_recipe'], value['job'], indices, 8)
        old_complete = dict(complete, recipe=ssl.FIT_LABEL_RECIPE)
        for key in ('fit_sampling_policy', 'fit_global_positive_quota', 'fit_positive_primary_draws',
                'fit_negative_primary_draws', 'contrastive_fit_positive_images'): old_complete.pop(key)
        self.assertTrue(ssl.ssl_phase_completed(old_complete, ssl.FIT_LABEL_RECIPE))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']))
    torch.set_num_interop_threads(1)
    unittest.main()
