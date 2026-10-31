"""Two synthetic checks for the R128 head/age-only warmup policy.

Parent A runs these once on the guarded Pod.  A tiny optimizer-backed model is
stepped from the committed cursor; no real Small model, images or assets.
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import age, engine, head_warmup
from retrain.persistence import canonical_digest

ROLES = ('backbone', 'head', 'age')


def age_spec():
    contract = {'version': age.VERSION, 'field': 'age', 'valid_min_exclusive': 0,
                'valid_max_inclusive': 120, 'features': ['standardized_age', 'missing_indicator'],
                'parameter_count': 2, 'bias': False, 'lr': 1e-3, 'weight_decay': 1e-4,
                'fit_count': 4, 'valid_fit_count': 4, 'missing_fit_count': 0,
                'median': 50.0, 'mean': 50.0, 'population_std': 10.0}
    return {'enabled': True, 'contract': contract, 'contract_sha256': canonical_digest(contract)}


def job(spec):
    return {'head_lr': 1e-3, 'backbone_lr': 1e-4, 'weight_decay': 1e-4, 'age': spec}


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 5, bias=False)
        self.classifier = nn.Linear(5, 1)


def assign(model):
    model.features.weight.grad = torch.full((5, 4), .5)
    model.classifier.weight.grad = torch.full((1, 5), 3.)
    model.classifier.bias.grad = torch.full((1,), 3.)
    model.age_weight.grad = torch.tensor([2., -2.])


def reference_mask_clip(groups, successful_updates):
    next_update = successful_updates + 1
    active = ['head', 'age'] if next_update <= 1600 else list(ROLES)
    pieces = [p.grad.detach().float().reshape(-1) for role in active
              for p in groups[role] if p.grad is not None]
    total = torch.linalg.vector_norm(torch.cat(pieces)) if pieces else torch.zeros(())
    coefficient = torch.clamp(1.0 / (total + 1e-6), max=1.0)
    if next_update <= 1600:
        for parameter in groups['backbone']:
            parameter.grad = None
    for role in active:
        for parameter in groups[role]:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)


class HeadWarmupTest(unittest.TestCase):
    def test_warmup_steps_backbone_freeze_and_reference_adam(self):
        spec = age_spec()
        torch.manual_seed(41)
        model = Toy()
        torch.manual_seed(41)
        reference = Toy()
        age.attach_age(model, spec)
        age.attach_age(reference, spec)
        optimizer = engine._optimizer(model, job(spec))
        reference_optimizer = engine._optimizer(reference, job(spec))
        metadata = [(g['role'], g['lr'], g['weight_decay'], [id(p) for p in g['params']])
                    for g in optimizer.param_groups]
        self.assertEqual([(role, lr, wd) for role, lr, wd, _ in metadata],
                         [('backbone', 1e-4, 1e-4), ('head', 1e-3, 0.0),
                          ('head', 1e-3, 1e-4), ('age', 1e-3, 1e-4)])
        self.assertEqual(optimizer.state_dict()['state'], {})

        assign(model)
        rng = torch.random.get_rng_state()
        saved_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
        self.assertIsNone(head_warmup.build_update_groups(optimizer, None))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(optimizer.state_dict()['state'], {})
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter.grad, saved_grads[name]))

        groups = head_warmup.build_update_groups(optimizer, head_warmup.HEAD_WARMUP_RULE)
        self.assertEqual(set(groups), set(ROLES))
        self.assertEqual({id(p) for role in ROLES for p in groups[role]},
                         {id(p) for p in model.parameters()})
        self.assertEqual([(g['role'], g['lr'], g['weight_decay'], [id(p) for p in g['params']])
                          for g in optimizer.param_groups], metadata)
        self.assertEqual(head_warmup.expected_optimizer_steps(0),
                         {'backbone': None, 'head': None, 'age': None})
        self.assertEqual(head_warmup.expected_optimizer_steps(1),
                         {'backbone': None, 'head': 1, 'age': 1})
        self.assertEqual(head_warmup.expected_optimizer_steps(1600),
                         {'backbone': None, 'head': 1600, 'age': 1600})
        self.assertEqual(head_warmup.expected_optimizer_steps(1601),
                         {'backbone': 1, 'head': 1601, 'age': 1601})

        ref_groups = {role: [] for role in ROLES}
        for group in reference_optimizer.param_groups:
            ref_groups[group['role']].extend(group['params'])
        initial_backbone = model.features.weight.detach().clone()
        head_norm = (54.0) ** .5
        age_norm = 8 ** .5
        expected_coefficient = 1.0 / ((head_norm ** 2 + age_norm ** 2) ** .5 + 1e-6)
        names = {id(p): name for name, p in model.named_parameters()}
        reference_names = {id(p): name for name, p in reference.named_parameters()}
        for step in range(1601):
            assign(model)
            assign(reference)
            ok, info = head_warmup.mask_and_clip_gradients_(groups, step)
            self.assertTrue(ok)
            if step == 0:
                self.assertEqual(info['phase'], 'warmup')
                self.assertEqual(info['active_roles'], ['head', 'age'])
                self.assertEqual(info['next_successful_update'], 1)
                self.assertIsNone(model.features.weight.grad)
                self.assertAlmostEqual(float(info['clip_coefficient']), expected_coefficient, delta=1e-6)
                self.assertNotAlmostEqual(float(info['clip_coefficient']),
                                          1.0 / (54.0 ** .5 + 1e-6), delta=1e-4)
                self.assertAlmostEqual(float(model.classifier.weight.grad[0, 0]),
                                       3.0 * expected_coefficient, delta=1e-5)
                self.assertAlmostEqual(float(model.classifier.bias.grad[0]),
                                       3.0 * expected_coefficient, delta=1e-5)
                self.assertAlmostEqual(float(model.age_weight.grad[0]),
                                       2.0 * expected_coefficient, delta=1e-5)
            if step == 1599:
                self.assertEqual(info['phase'], 'warmup')
                self.assertEqual(info['next_successful_update'], 1600)
            if step == 1600:
                self.assertEqual(info['phase'], 'full')
                self.assertEqual(info['active_roles'], list(ROLES))
                self.assertIsNotNone(model.features.weight.grad)
            reference_mask_clip(ref_groups, step)
            optimizer.step()
            reference_optimizer.step()
            if step in (0, 1599, 1600):
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                               rtol=1e-5, atol=1e-7)
                actual_state = {names[id(p)]: s for p, s in optimizer.state.items()}
                reference_state = {reference_names[id(p)]: s
                                   for p, s in reference_optimizer.state.items()}
                self.assertEqual(set(actual_state), set(reference_state))
                for name in actual_state:
                    for field in ('exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(actual_state[name][field],
                                                   reference_state[name][field],
                                                   rtol=1e-5, atol=1e-7)
                    self.assertEqual(int(actual_state[name]['step']),
                                     int(reference_state[name]['step']))
                if step < 1600:
                    self.assertNotIn('features.weight', actual_state)
                    self.assertTrue(torch.equal(model.features.weight, initial_backbone))
                    self.assertEqual(int(actual_state['classifier.weight']['step']), step + 1)
                    self.assertEqual(int(actual_state['age_weight']['step']), step + 1)
                else:
                    self.assertEqual(int(actual_state['features.weight']['step']), 1)
                    self.assertEqual(int(actual_state['classifier.weight']['step']), 1601)
                    self.assertEqual(int(actual_state['age_weight']['step']), 1601)
                    self.assertFalse(torch.equal(model.features.weight, initial_backbone))

    def test_nonfinite_and_world_rejection_leave_everything_untouched(self):
        backbone = nn.Parameter(torch.zeros(2))
        head_decay = nn.Parameter(torch.zeros(2))
        head_no_decay = nn.Parameter(torch.zeros(2))
        age_weight = nn.Parameter(torch.zeros(2))
        optimizer = SimpleNamespace(param_groups=[
            {'role': 'backbone', 'params': [backbone]},
            {'role': 'head', 'params': [head_decay]},
            {'role': 'head', 'params': [head_no_decay]},
            {'role': 'age', 'params': [age_weight]}])
        groups = head_warmup.build_update_groups(optimizer, head_warmup.HEAD_WARMUP_RULE)

        backbone.grad = torch.tensor([.3, .4])
        head_decay.grad = torch.tensor([2., 0.])
        head_no_decay.grad = torch.tensor([0., 3.])
        age_weight.grad = torch.tensor([.2, 0.])
        originals = {id(p): p.grad for p in (backbone, head_decay, head_no_decay, age_weight)}
        backbone.grad[0] = float('nan')
        calls = []
        ok, info = head_warmup.mask_and_clip_gradients_(
            groups, 0, any_bad=lambda bad: calls.append(bad) or bad)
        self.assertFalse(ok)
        self.assertEqual(calls, [True])
        self.assertIsNone(info['clip_coefficient'])
        self.assertIsNotNone(backbone.grad)
        for parameter in (backbone, head_decay, head_no_decay, age_weight):
            self.assertIs(parameter.grad, originals[id(parameter)])
        self.assertTrue(bool(torch.isnan(backbone.grad[0])))
        self.assertTrue(torch.equal(head_decay.grad, torch.tensor([2., 0.])))

        backbone.grad = torch.tensor([.3, .4])
        backbone.grad[0] = float('inf')
        calls = []
        ok, info = head_warmup.mask_and_clip_gradients_(
            groups, 0, any_bad=lambda bad: calls.append(bad) or bad)
        self.assertFalse(ok)
        self.assertEqual(calls, [True])
        self.assertIsNotNone(backbone.grad)
        self.assertTrue(bool(torch.isinf(backbone.grad[0])))
        self.assertTrue(torch.equal(head_decay.grad, torch.tensor([2., 0.])))

        backbone.grad = torch.tensor([.3, .4])
        calls = []
        ok, info = head_warmup.mask_and_clip_gradients_(
            groups, 0, any_bad=lambda bad: calls.append(bad) or True)
        self.assertFalse(ok)
        self.assertEqual(calls, [False])
        self.assertIsNotNone(backbone.grad)
        self.assertTrue(torch.equal(head_decay.grad, torch.tensor([2., 0.])))
        self.assertTrue(torch.equal(age_weight.grad, torch.tensor([.2, 0.])))

        age_weight.grad = torch.tensor([float('inf'), 0.])
        ok, info = head_warmup.mask_and_clip_gradients_(groups, 0)
        self.assertFalse(ok)
        self.assertIsNotNone(backbone.grad)
        self.assertTrue(torch.equal(head_decay.grad, torch.tensor([2., 0.])))

        # Targeted mock of the norm-aggregation boundary (not a measured
        # overflow): three role norms are finite while the active total is not.
        backbone.grad = torch.tensor([.3, .4])
        head_decay.grad = torch.tensor([2., 0.])
        head_no_decay.grad = torch.tensor([0., 3.])
        age_weight.grad = torch.tensor([.2, 0.])
        mocked = {id(p): p.grad for p in (backbone, head_decay, head_no_decay, age_weight)}
        mocked_values = {id(p): p.grad.clone() for p in (backbone, head_decay, head_no_decay, age_weight)}
        calls = []
        finite_norm = torch.tensor(1.0)
        infinite_norm = torch.tensor(float('inf'))
        with patch.object(head_warmup, '_norm',
                          side_effect=[finite_norm, finite_norm, finite_norm, infinite_norm]):
            ok, info = head_warmup.mask_and_clip_gradients_(
                groups, 0, any_bad=lambda bad: calls.append(bad) or bad)
        self.assertFalse(ok)
        self.assertEqual(calls, [True])
        self.assertTrue(all(bool(torch.isfinite(info['role_norms'][role])) for role in ROLES))
        self.assertFalse(bool(torch.isfinite(info['active_global_norm'])))
        for parameter in (backbone, head_decay, head_no_decay, age_weight):
            self.assertIs(parameter.grad, mocked[id(parameter)])
            self.assertTrue(torch.equal(parameter.grad, mocked_values[id(parameter)]))
        self.assertTrue(torch.equal(head_decay.grad, torch.tensor([2., 0.])))

        backbone.grad = torch.tensor([.3, .4])
        age_weight.grad = torch.tensor([.2, 0.])
        ok, info = head_warmup.mask_and_clip_gradients_(groups, 0)
        self.assertTrue(ok)
        self.assertEqual(info['active_roles'], ['head', 'age'])
        self.assertIsNone(backbone.grad)
        combined = (13.0 + .04) ** .5
        coefficient = 1.0 / (combined + 1e-6)
        self.assertAlmostEqual(float(info['clip_coefficient']), coefficient, delta=1e-6)
        self.assertAlmostEqual(float(head_decay.grad[0]), 2.0 * coefficient, delta=1e-5)
        self.assertAlmostEqual(float(head_no_decay.grad[1]), 3.0 * coefficient, delta=1e-5)
        self.assertAlmostEqual(float(age_weight.grad[0]), .2 * coefficient, delta=1e-5)

        backbone.grad = torch.tensor([.3, .4])
        head_decay.grad = torch.tensor([2., 0.])
        head_no_decay.grad = torch.tensor([0., 3.])
        age_weight.grad = torch.tensor([.2, 0.])
        ok, info = head_warmup.mask_and_clip_gradients_(groups, 1600)
        self.assertTrue(ok)
        self.assertEqual(info['phase'], 'full')
        self.assertIsNotNone(backbone.grad)
        combined = (13.0 + .04 + .25) ** .5
        coefficient = 1.0 / (combined + 1e-6)
        self.assertAlmostEqual(float(backbone.grad[0]), .3 * coefficient, delta=1e-5)
        self.assertAlmostEqual(float(head_decay.grad[0]), 2.0 * coefficient, delta=1e-5)
        self.assertAlmostEqual(float(age_weight.grad[0]), .2 * coefficient, delta=1e-5)
        self.assertAlmostEqual(float(age_weight.grad[1]), 0.0, delta=1e-8)


if __name__ == '__main__':
    unittest.main()
