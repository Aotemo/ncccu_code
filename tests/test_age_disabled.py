"""Two synthetic checks for the R123 age-disabled branch.

Parent A runs these once on the guarded Pod.  They exercise the existing
attach_age / engine._optimizer / Dataset / unpack_batch / forward paths with a
toy classifier, never the production model or real assets.
"""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import age, data
from retrain.persistence import canonical_digest

BaseRandom = random.Random
SEED = 20261029


def spec(enabled):
    """Smallest spec accepted by age.validate_spec; only the flag differs."""
    contract = {'version': age.VERSION, 'field': 'age', 'valid_min_exclusive': 0,
                'valid_max_inclusive': 120, 'features': ['standardized_age', 'missing_indicator'],
                'parameter_count': 2, 'bias': False, 'lr': 1e-3, 'weight_decay': 1e-4,
                'fit_count': 4, 'valid_fit_count': 4, 'missing_fit_count': 0,
                'median': 50.0, 'mean': 50.0, 'population_std': 10.0}
    return {'enabled': enabled, 'contract': contract, 'contract_sha256': canonical_digest(contract)}


def toy_model():
    import torch
    from torch import nn

    class ToyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Linear(4, 5, bias=False)
            self.classifier = nn.Linear(5, 1)

        def forward(self, images, age_features=None):
            logits = self.classifier(torch.relu(self.features(images))).reshape(-1)
            return age.residual(self, logits, age_features) if age_features is not None else logits

    return ToyClassifier()


class AgeDisabledTest(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'train').mkdir()
        self.image = Image.new('RGB', (40, 28))
        self.image.putdata([(20 + x * 5 % 220, 15 + y * 7 % 230, 25 + (x * y) % 210)
                            for y in range(28) for x in range(40)])
        self.row = {'name': 'synthetic.png', 'relative_path': 'train/synthetic.png',
                    'sha256': 'a' * 64, 'label': 1, 'age': '51'}

    def make(self, age_spec, training=True, include_labels=True):
        with patch.object(data, 'configured_cache', return_value=None):
            return data.ImageDataset(self.root, [self.row], 32, training,
                                     age=age_spec, include_labels=include_labels)

    def render(self, dataset, seed=SEED):
        rngs = []

        def factory(value):
            rng = BaseRandom(value)
            rngs.append(rng)
            return rng

        key = (0, seed) if dataset.training else 0
        with patch.object(dataset, '_read_image', return_value=self.image.copy()), \
                patch.object(data.random, 'Random', side_effect=factory):
            result = dataset[key]
        return result, rngs

    def test_disabled_age_has_no_parameter_role_or_rng_change(self):
        import torch
        from retrain import engine
        enabled_spec, disabled_spec = spec(True), spec(False)
        torch.manual_seed(11)
        with_age = toy_model()
        torch.manual_seed(11)
        without_age = toy_model()
        initial = {key: value.detach().clone() for key, value in with_age.state_dict().items()}

        state = torch.random.get_rng_state()
        age.attach_age(with_age, enabled_spec)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        state = torch.random.get_rng_state()
        age.attach_age(without_age, disabled_spec)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

        self.assertEqual(set(with_age.state_dict()), set(initial) | {'age_weight'})
        self.assertEqual(set(without_age.state_dict()), set(initial))
        for key, value in initial.items():
            self.assertTrue(torch.equal(with_age.state_dict()[key], value))
            self.assertTrue(torch.equal(without_age.state_dict()[key], value))
        self.assertEqual(tuple(with_age.age_weight.shape), (2,))
        self.assertEqual(with_age.age_weight.dtype, torch.float32)
        self.assertTrue(bool((with_age.age_weight == 0).all()))
        self.assertFalse(hasattr(without_age, 'age_weight'))

        images = torch.tensor([[.1, -.2, .3, .4], [.5, .6, -.7, .8], [-.1, .2, 0., .3]])
        self.assertTrue(torch.equal(with_age(images), without_age(images)))
        self.assertEqual(tuple(with_age(images).shape), (3,))

        job_enabled = {'head_lr': 1e-3, 'backbone_lr': 1e-4, 'weight_decay': 1e-4,
                       'age': enabled_spec}
        job_disabled = {'head_lr': 1e-3, 'backbone_lr': 1e-4, 'weight_decay': 1e-4,
                        'age': disabled_spec}
        optimizer_enabled = engine._optimizer(with_age, job_enabled)
        optimizer_disabled = engine._optimizer(without_age, job_disabled)
        disabled_groups = optimizer_disabled.param_groups
        self.assertEqual({group['role'] for group in disabled_groups}, {'backbone', 'head'})
        self.assertEqual({id(p) for group in disabled_groups for p in group['params']},
                         {id(p) for p in without_age.parameters()})
        self.assertEqual(optimizer_disabled.state_dict()['state'], {})

        age_groups = [group for group in optimizer_enabled.param_groups if group['role'] == 'age']
        self.assertEqual(len(age_groups), 1)
        self.assertEqual(sum(p.numel() for p in age_groups[0]['params']), 2)
        self.assertEqual(age_groups[0]['lr'], 1e-3)
        self.assertEqual(age_groups[0]['weight_decay'], 1e-4)
        self.assertEqual({id(p) for group in optimizer_enabled.param_groups for p in group['params']},
                         {id(p) for p in with_age.parameters()})
        self.assertEqual(optimizer_enabled.state_dict()['state'], {})

    def test_dataset_and_forward_skip_age_values_when_disabled(self):
        import torch
        enabled_spec, disabled_spec = spec(True), spec(False)
        with_age, without_age = self.make(enabled_spec), self.make(disabled_spec)

        with patch.object(age, 'features', return_value=[.25, 0.]) as features:
            enabled_result, enabled_rng = self.render(with_age)
        features.assert_called_once_with(self.row, enabled_spec)
        with patch.object(age, 'features',
                          side_effect=AssertionError('disabled path must not read age')) as unused:
            disabled_result, disabled_rng = self.render(without_age)
        unused.assert_not_called()

        self.assertEqual(len(enabled_result), 4)
        self.assertEqual(len(disabled_result), 3)
        self.assertTrue(torch.equal(enabled_result[0], disabled_result[0]))
        self.assertEqual(enabled_result[1:3], disabled_result[1:3])
        self.assertEqual(enabled_result[3].tolist(), [.25, 0.])
        self.assertEqual(len(disabled_rng), len(enabled_rng))
        self.assertEqual(disabled_rng[0].getstate(), enabled_rng[0].getstate())
        self.assertEqual(disabled_rng[0].random(), enabled_rng[0].random())

        baseline = self.make(None, training=False)
        disabled_eval = self.make(disabled_spec, training=False)
        baseline_result, baseline_rng = self.render(baseline)
        eval_result, eval_rng = self.render(disabled_eval)
        self.assertTrue(torch.equal(baseline_result[0], eval_result[0]))
        self.assertEqual((baseline_rng, eval_rng), ([], []))

        images = torch.tensor([[.1, -.2, .3, .4], [.5, .6, -.7, .8], [-.1, .2, 0., .3]])
        labels, indices = torch.tensor([0., 1., 0.]), torch.tensor([2, 5, 7])
        age_rows = torch.tensor([[.5, 0.], [-1.5, 1.], [2., 0.]])
        job_enabled = {'age': enabled_spec}
        job_disabled = {'age': disabled_spec}

        batch_disabled = age.unpack_batch((images, labels, indices), job_disabled, 'cpu')
        self.assertIsNone(batch_disabled[3])
        torch.manual_seed(11)
        disabled_model = toy_model()
        age.attach_age(disabled_model, disabled_spec)
        # A NaN age row is ignored when the branch is disabled.
        ignored = torch.full_like(age_rows, float('nan'))
        disabled_logits = age.forward(disabled_model, batch_disabled[0], ignored, job_disabled)
        self.assertTrue(torch.equal(disabled_logits, disabled_model(images)))

        torch.manual_seed(11)
        enabled_model = toy_model()
        age.attach_age(enabled_model, enabled_spec)
        with torch.no_grad():
            enabled_model.age_weight.copy_(torch.tensor([.5, -.25]))
        batch_enabled = age.unpack_batch((images, labels, indices, age_rows), job_enabled, 'cpu')
        enabled_logits = age.forward(enabled_model, batch_enabled[0], batch_enabled[3], job_enabled)
        expected = enabled_model(images) + (batch_enabled[3] * enabled_model.age_weight).sum(dim=1)
        self.assertTrue(torch.equal(enabled_logits, expected))
        self.assertFalse(torch.equal(enabled_logits, enabled_model(images)))


if __name__ == '__main__':
    unittest.main()
