"""Independent fit-label loss oracles and original-stream/fit-only contracts."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


class FitLabelContrastiveTest(unittest.TestCase):
    def test_outside_log_oracle_gradients_and_instance_limit(self):
        import torch
        from retrain import vicreg
        self.assertTrue(callable(getattr(vicreg, 'supervised_contrastive_loss', None)),
            'fixed fit-label contrastive loss missing')
        a = torch.tensor([[1., 2., -.5], [.3, -1., 2.], [2., .5, 1.]],
            dtype=torch.float64, requires_grad=True)
        b = torch.tensor([[1.2, 1.6, -.2], [.1, -1.2, 1.8], [1.7, .3, 1.3]],
            dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 0, 1])
        vectors = a.detach().tolist() + b.detach().tolist()
        normalized = [[x / math.sqrt(sum(y*y for y in row)) for x in row] for row in vectors]
        repeated = labels.tolist() * 2
        terms = []
        for i, row in enumerate(normalized):
            scores = {j: sum(x*y for x, y in zip(row, other)) / .1
                for j, other in enumerate(normalized) if j != i}
            peak = max(scores.values())
            normalizer = peak + math.log(sum(math.exp(x-peak) for x in scores.values()))
            targets = [v for j, v in scores.items() if repeated[j] == repeated[i]]
            terms.append(normalizer - sum(targets) / len(targets))
        loss, stats = vicreg.supervised_contrastive_loss(a, b, labels, temperature=.1)
        self.assertAlmostEqual(loss.item(), sum(terms)/len(terms), places=12)
        self.assertEqual(stats['contrastive_logits_shape'], [6, 6])
        self.assertEqual(stats['contrastive_positive_targets_per_anchor_min'], 1)
        self.assertEqual(stats['contrastive_positive_targets_per_anchor_max'], 3)
        self.assertEqual(stats['contrastive_fit_positive_images'], 1)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, y: vicreg.supervised_contrastive_loss(x, y, labels, temperature=.1)[0],
            (a, b), eps=1e-6, atol=1e-5))
        unique = torch.arange(3)
        ordinary = vicreg.nt_xent_loss(a, b, temperature=.1)[0]
        self.assertTrue(torch.allclose(ordinary,
            vicreg.supervised_contrastive_loss(a, b, unique, temperature=.1)[0], atol=1e-12, rtol=0))
        permutation = torch.tensor([2, 0, 1])
        for x, y, tag in ((b, a, labels), (a[permutation], b[permutation], labels[permutation]),
                (a*2., b*3., labels)):
            self.assertTrue(torch.allclose(loss,
                vicreg.supervised_contrastive_loss(x, y, tag, temperature=.1)[0], atol=1e-12, rtol=0))
        identical = torch.ones((3, 4), dtype=torch.float64, requires_grad=True)
        same = vicreg.supervised_contrastive_loss(identical, identical, torch.zeros(3, dtype=torch.long), temperature=.1)[0]
        self.assertAlmostEqual(same.item(), math.log(5), places=12)
        self.assertTrue(torch.isfinite(torch.autograd.grad(same, identical)[0]).all())

    def test_fit_label_access_recipe_and_original_supervision(self):
        import torch
        from retrain import ssl_stage as ssl, experiment91 as ex
        self.assertTrue(hasattr(ssl, 'FIT_LABEL_RECIPE'), 'fixed fit-label recipe missing')
        self.assertTrue(callable(getattr(ssl, 'fit_label_table', None)), 'fit-only label table missing')
        expected = dict(ssl.STRONG_COLOR_RECIPE, version='fit-label-supcon-1600-v1',
            loss='supervised_contrastive_out', label_source='fit_csv')
        self.assertEqual(ssl.FIT_LABEL_RECIPE, expected)
        complete = dict(status='completed', recipe=expected, successful_updates=1600,
            attempted_updates=1600, skipped_updates=0, dev_evaluations=0,
            labels_or_age_used_for_loss=True, labels_used_for_loss=True,
            age_used_for_loss=False, label_scope='fit_csv')
        self.assertTrue(ssl.ssl_phase_completed(complete, expected))
        for changed in ({'labels_used_for_loss': False}, {'age_used_for_loss': True},
                {'label_scope': 'dev_csv'}, {'labels_or_age_used_for_loss': False}):
            self.assertFalse(ssl.ssl_phase_completed(dict(complete, **changed), expected))
        class FitRow:
            def __init__(self, value): self.value = value
            def __getitem__(self, key):
                if key != 'label': raise AssertionError('age or other row field accessed')
                return self.value
        class DevRow:
            def __getitem__(self, key): raise AssertionError('dev label accessed')
        manifest = {'rows': [FitRow(0), DevRow(), FitRow(1.)],
            'fit_indices': [2, 0], 'dev_indices': [1], 'audit_indices': []}
        table = ssl.fit_label_table(manifest)
        self.assertEqual(table, {2: 1, 0: 0})
        actual = ssl.gather_fit_labels(table, torch.tensor([2, 0]), device='cpu')
        self.assertTrue(torch.equal(actual, torch.tensor([1, 0])))
        with self.assertRaises(ValueError): ssl.gather_fit_labels(table, [1], device='cpu')
        for changed in (dict(manifest, fit_indices=[2, 2]), dict(manifest, dev_indices=[0, 1]),
                dict(manifest, rows=[FitRow(.5), DevRow(), FitRow(1)])):
            with self.assertRaises(ValueError): ssl.fit_label_table(changed)
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        for label, seed, order, binding in (
            ('R238', 20261029, 153, ex.FIT_LABEL_CONTRASTIVE_SMALL_BINDING),
            ('R239', 20261027, 154, ex.FIT_LABEL_CONTRASTIVE_SMALL_SEED27_BINDING)):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            job = copy.deepcopy(original['job']); job['seed'] = seed
            self.assertEqual(value['job'], job)
            self.assertEqual(value['candidate_profile'], binding)
            self.assertEqual(value['candidate']['experiment_order'], order)
            self.assertEqual(value['ssl_recipe'], expected)
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            for key in ('public_source', 'paths', 'data_split', 'schedule', 'early_stopping', 'structure'):
                self.assertEqual(value[key], original[key])
            current = ssl.ssl_rng_derivation(expected, seed, 8)
            prior = ssl.ssl_rng_derivation(ssl.STRONG_COLOR_RECIPE, seed, 8)
            for key in ('learning_domain', 'sampling_domain', 'projector_seed', 'model_seeds', 'sampler_seed', 'loader_seeds'):
                self.assertEqual(current[key], prior[key])
            indices = {'fit_indices': list(range(128)), 'dev_indices': [128], 'audit_indices': []}
            self.assertEqual(ssl.validate_recipe(expected, job, indices, 8), expected)
            with self.assertRaises(ValueError): ssl.validate_recipe(dict(expected, label_source='dev_csv'), job, indices, 8)
        for label in ('R115', 'R228', 'R230', 'R232', 'R234', 'R235', 'R236', 'R237'):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            ssl.validate_recipe(value['ssl_recipe'], value['job'], indices, 8)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']))
    torch.set_num_interop_threads(1)
    unittest.main()
