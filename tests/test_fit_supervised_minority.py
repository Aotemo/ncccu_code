"""Independent loss/gradient and frozen-contract checks for fit Supervised Minority."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


class FitSupervisedMinorityTest(unittest.TestCase):
    def test_minority_positive_sets_loss_gradient_and_old_default(self):
        import torch
        from retrain.vicreg import supervised_contrastive_loss, nt_xent_loss
        gen = torch.Generator().manual_seed(824268)
        a = torch.randn(5, 4, generator=gen, dtype=torch.float64, requires_grad=True)
        b = torch.randn(5, 4, generator=gen, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 0])

        def scalar_oracle(first, second, tags, minority_only):
            vectors = [row / row.square().sum().sqrt() for row in [*first, *second]]
            losses = []
            n = len(tags)
            for i, anchor in enumerate(vectors):
                similarities = [(anchor * other).sum() / .1 for other in vectors]
                denominator = torch.logsumexp(torch.stack([s for j, s in enumerate(similarities) if i != j]), 0)
                positives = [j for j in range(2*n) if j != i and
                    (j % n == i % n if minority_only and int(tags[i % n]) == 0
                     else int(tags[j % n]) == int(tags[i % n]))]
                losses.append(denominator - sum(similarities[j] for j in positives) / len(positives))
            return torch.stack(losses).mean()

        expected = scalar_oracle(a, b, labels, True)
        loss, stats = supervised_contrastive_loss(a, b, labels, temperature=.1,
            majority_instance_only=True)
        torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
        actual_grad = torch.autograd.grad(loss, (a, b), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (a, b), retain_graph=True)
        for actual, wanted in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, wanted, rtol=1e-11, atol=1e-11)
        old, old_stats = supervised_contrastive_loss(a, b, labels, temperature=.1)
        torch.testing.assert_close(old, scalar_oracle(a, b, labels, False), rtol=1e-12, atol=1e-12)
        self.assertEqual(old_stats['contrastive_target'], 'fit_csv_same_label')
        self.assertEqual(stats['contrastive_target'], 'fit_csv_minority_same_label_majority_same_image')
        self.assertEqual(stats['contrastive_minority_label'], 1)
        self.assertEqual(stats['contrastive_positive_targets_per_anchor_min'], 1)
        self.assertEqual(stats['contrastive_positive_targets_per_anchor_max'], 3)
        for tags, comparison in ((torch.zeros(5, dtype=torch.long), nt_xent_loss(a, b, temperature=.1)[0]),
                (torch.ones(5, dtype=torch.long), supervised_contrastive_loss(a, b, torch.ones(5, dtype=torch.long), temperature=.1)[0])):
            actual = supervised_contrastive_loss(a, b, tags, temperature=.1, majority_instance_only=True)[0]
            torch.testing.assert_close(actual, comparison, rtol=1e-12, atol=1e-12)
        order = torch.tensor([4, 2, 1, 0, 3])
        permuted = supervised_contrastive_loss(a[order], b[order], labels[order], temperature=.1, majority_instance_only=True)[0]
        swapped = supervised_contrastive_loss(b, a, labels, temperature=.1, majority_instance_only=True)[0]
        torch.testing.assert_close(loss, permuted, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(loss, swapped, rtol=1e-12, atol=1e-12)
        x, y = torch.randn(64, 7, generator=gen), torch.randn(64, 7, generator=gen)
        _, actual = supervised_contrastive_loss(x, y, torch.tensor([1]*8 + [0]*56), temperature=.1, majority_instance_only=True)
        for key, expected in {'contrastive_logits_shape': [128, 128], 'contrastive_anchor_count': 128,
                'contrastive_positive_targets_per_anchor_min': 1, 'contrastive_positive_targets_per_anchor_max': 15,
                'contrastive_negative_views_per_anchor_min': 112, 'contrastive_negative_views_per_anchor_max': 126,
                'contrastive_fit_positive_images': 8, 'contrastive_self_excluded': True}.items():
            self.assertEqual(actual[key], expected)
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(a, b, torch.tensor([0, 1, 2, 1, 0]), temperature=.1, majority_instance_only=True)

    def test_recipe_completion_original_supervision_and_streams(self):
        from retrain import ssl_stage as ssl, experiment91 as ex
        self.assertTrue(hasattr(ssl, 'FIT_SUPMIN_RECIPE'), 'fixed fit Supervised Minority recipe missing')
        expected = dict(ssl.FIT_LABEL_QUOTA_RECIPE, version='fit-label-supminquota8-1600-v1',
            loss='supervised_minority_out', positive_policy='minority_same_label_majority_same_image', minority_label=1)
        self.assertEqual(ssl.FIT_SUPMIN_RECIPE, expected)
        complete = dict(status='completed', recipe=expected, successful_updates=1600,
            attempted_updates=1600, skipped_updates=0, dev_evaluations=0,
            labels_or_age_used_for_loss=True, labels_used_for_loss=True, age_used_for_loss=False, label_scope='fit_csv',
            fit_sampling_policy='fit-class-quota-g64-p8-v1', fit_global_positive_quota=8,
            fit_positive_primary_draws=12800, fit_negative_primary_draws=89600, contrastive_fit_positive_images=8,
            contrastive_target='fit_csv_minority_same_label_majority_same_image', contrastive_minority_label=1,
            contrastive_positive_targets_per_anchor_min=1, contrastive_positive_targets_per_anchor_max=15,
            contrastive_negative_views_per_anchor_min=112, contrastive_negative_views_per_anchor_max=126)
        self.assertTrue(ssl.ssl_phase_completed(complete, expected))
        for changed in ({'contrastive_target': 'fit_csv_same_label'}, {'contrastive_minority_label': 0},
                {'contrastive_positive_targets_per_anchor_max': 111}, {'contrastive_negative_views_per_anchor_min': 16},
                {'fit_positive_primary_draws': 12799}, {'age_used_for_loss': True}, {'label_scope': 'dev_csv'}):
            self.assertFalse(ssl.ssl_phase_completed(dict(complete, **changed), expected))
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        manifest = {'fit_indices': list(range(128)), 'dev_indices': [128], 'audit_indices': []}
        for label, prior_label, seed, order, binding in (
                ('R242', 'R240', 20261029, 157, ex.FIT_SUPMIN_SMALL_BINDING),
                ('R243', 'R241', 20261027, 158, ex.FIT_SUPMIN_SMALL_SEED27_BINDING)):
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
            left, right = ssl.ssl_rng_derivation(expected, seed, 8), ssl.ssl_rng_derivation(prior['ssl_recipe'], seed, 8)
            for key in ('learning_domain', 'sampling_domain', 'projector_seed', 'model_seeds', 'sampler_seed', 'loader_seeds'):
                self.assertEqual(left[key], right[key])
            self.assertEqual(ssl.validate_recipe(expected, job, manifest, 8), expected)
        old_complete = dict(complete, recipe=ssl.FIT_LABEL_QUOTA_RECIPE, contrastive_target='fit_csv_same_label')
        old_complete.pop('contrastive_minority_label')
        self.assertTrue(ssl.ssl_phase_completed(old_complete, ssl.FIT_LABEL_QUOTA_RECIPE))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(max(1,int(os.environ['NCCCU_CPU_THREADS'])-2)); torch.set_num_interop_threads(1)
    unittest.main()
