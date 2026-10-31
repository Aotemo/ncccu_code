"""Two synthetic checks for the R126 rolewise L2 clip policy.

Parent A runs these once on the guarded Pod.  No real Small model, weights,
images or forward benchmark are used; the age contract is the same minimal
legal spec already used by the age tests.
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import age, ema, engine, gradient_clip
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


def tensor_parameter(values, gradient):
    parameter = nn.Parameter(torch.tensor(values, dtype=torch.float32))
    parameter.grad = torch.tensor(gradient, dtype=torch.float32)
    return parameter


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 5, bias=False)
        self.classifier = nn.Linear(5, 1)

    def forward(self, images, age_features):
        logits = self.classifier(torch.relu(self.features(images))).flatten()
        return logits + (age_features * self.age_weight).sum(dim=1)


def reference_groups(optimizer):
    groups = {role: [] for role in ROLES}
    for group in optimizer.param_groups:
        groups[group['role']].extend(group['params'])
    return groups


def reference_clip(groups):
    norms = {}
    for role, parameters in groups.items():
        pieces = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
        norms[role] = torch.linalg.vector_norm(torch.cat(pieces)) if pieces else torch.zeros(())
    coefficients = {role: min(1.0, 1.0 / (float(norms[role]) + 1e-6)) for role in groups}
    for role, parameters in groups.items():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficients[role])
    return coefficients


class RolewiseClipTest(unittest.TestCase):
    def test_hand_vectors_merge_roles_and_nonfinite_precheck(self):
        backbone = tensor_parameter([0., 0.], [.3, .4])
        head_decay = tensor_parameter([0., 0.], [2., 0.])
        head_no_decay = tensor_parameter([0., 0.], [0., 3.])
        age_weight = tensor_parameter([0., 0.], [.2, 0.])
        optimizer = SimpleNamespace(param_groups=[
            {'role': 'backbone', 'params': [backbone]},
            {'role': 'head', 'params': [head_decay]},
            {'role': 'head', 'params': [head_no_decay]},
            {'role': 'age', 'params': [age_weight]}])
        state = torch.random.get_rng_state()
        saved_grads = [p.grad.clone() for p in (backbone, head_decay, head_no_decay, age_weight)]
        self.assertIsNone(gradient_clip.build_role_groups(optimizer, None))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for parameter, saved in zip((backbone, head_decay, head_no_decay, age_weight), saved_grads):
            self.assertTrue(torch.equal(parameter.grad, saved))

        groups = gradient_clip.build_role_groups(optimizer, gradient_clip.ROLEWISE_CLIP_RULE)
        self.assertEqual(set(groups), set(ROLES))
        self.assertEqual(groups['backbone'], [backbone])
        self.assertEqual(groups['head'], [head_decay, head_no_decay])
        self.assertEqual(groups['age'], [age_weight])
        norms = gradient_clip.measure_role_norms(groups)
        self.assertAlmostEqual(float(norms['backbone']), .5, places=6)
        self.assertAlmostEqual(float(norms['head']), 13 ** .5, places=6)
        self.assertAlmostEqual(float(norms['age']), .2, places=6)
        coefficients = gradient_clip.apply_role_clip_(groups, norms)
        self.assertEqual(float(coefficients['backbone']), 1.0)
        self.assertEqual(float(coefficients['age']), 1.0)
        expected = 1.0 / (13 ** .5 + 1e-6)
        self.assertAlmostEqual(float(coefficients['head']), expected, delta=1e-6)
        torch.testing.assert_close(head_decay.grad, torch.tensor([2., 0.]) * expected)
        torch.testing.assert_close(head_no_decay.grad, torch.tensor([0., 3.]) * expected)
        torch.testing.assert_close(backbone.grad, torch.tensor([.3, .4]))
        torch.testing.assert_close(age_weight.grad, torch.tensor([.2, 0.]))
        self.assertLessEqual(float(gradient_clip.measure_role_norms(groups)['head']), 1.0)

        broken = gradient_clip.build_role_groups(optimizer, gradient_clip.ROLEWISE_CLIP_RULE)
        head_no_decay.grad[1] = float('inf')
        before = {role: [p.grad.clone() for p in params] for role, params in broken.items()}
        broken_norms = gradient_clip.measure_role_norms(broken)
        self.assertFalse(bool(torch.isfinite(broken_norms['head'])))
        for role in ROLES:
            for gradient, saved in zip((p.grad for p in broken[role]), before[role]):
                self.assertTrue(torch.equal(gradient, saved))
        with self.assertRaises(ValueError):
            gradient_clip.apply_role_clip_(broken, broken_norms)
        for role in ROLES:
            for gradient, saved in zip((p.grad for p in broken[role]), before[role]):
                self.assertTrue(torch.equal(gradient, saved))

    def test_engine_roles_two_step_adamw_reference_and_state(self):
        spec = age_spec()
        torch.manual_seed(31)
        model = Toy()
        torch.manual_seed(31)
        reference = Toy()
        age.attach_age(model, spec)
        age.attach_age(reference, spec)
        with torch.no_grad():
            model.age_weight.copy_(torch.tensor([.3, -.2]))
            reference.age_weight.copy_(torch.tensor([.3, -.2]))
        torch.manual_seed(32)
        images = torch.randn(4, 4)
        age_rows = torch.tensor([[.5, 0.], [-1., 1.], [0., 1.], [1.5, 0.]])
        targets = torch.tensor([1., -1., .5, -.5])

        optimizer = engine._optimizer(model, job(spec))
        reference_optimizer = engine._optimizer(reference, job(spec))
        before = [(g['role'], g['lr'], g['weight_decay'], [id(p) for p in g['params']])
                  for g in optimizer.param_groups]
        self.assertEqual([(role, lr, wd) for role, lr, wd, _ in before],
                         [('backbone', 1e-4, 1e-4), ('head', 1e-3, 0.0),
                          ('head', 1e-3, 1e-4), ('age', 1e-3, 1e-4)])
        groups = gradient_clip.build_role_groups(optimizer, gradient_clip.ROLEWISE_CLIP_RULE)
        after = [(g['role'], g['lr'], g['weight_decay'], [id(p) for p in g['params']])
                 for g in optimizer.param_groups]
        self.assertEqual(before, after)
        self.assertEqual(optimizer.state_dict()['state'], {})
        self.assertEqual([g['role'] for g in optimizer.param_groups].count('head'), 2)
        all_ids = {id(p) for role in ROLES for p in groups[role]}
        self.assertEqual(all_ids, {id(p) for p in model.parameters()})
        self.assertEqual(sum(len(groups[role]) for role in ROLES), len(list(model.parameters())))

        ref_groups = reference_groups(reference_optimizer)
        names = {id(p): name for name, p in model.named_parameters()}
        reference_names = {id(p): name for name, p in reference.named_parameters()}
        actual_coefficients, reference_coefficients = [], []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(images, age_rows) - targets) ** 2).mean() * 200
            loss.backward()
            norms = gradient_clip.measure_role_norms(groups)
            actual_coefficients.append(gradient_clip.apply_role_clip_(groups, norms))
            self.assertTrue(any(float(value) < 1.0 for value in actual_coefficients[-1].values()))
            optimizer.step()

            reference_optimizer.zero_grad(set_to_none=True)
            reference_loss = ((reference(images, age_rows) - targets) ** 2).mean() * 200
            reference_loss.backward()
            reference_coefficients.append(reference_clip(ref_groups))
            reference_optimizer.step()

            for role in ROLES:
                self.assertAlmostEqual(float(actual_coefficients[-1][role]),
                                       reference_coefficients[-1][role], places=5)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                           rtol=1e-5, atol=1e-7)
            actual_state = {names[id(p)]: s for p, s in optimizer.state.items()}
            reference_state = {reference_names[id(p)]: s for p, s in reference_optimizer.state.items()}
            self.assertEqual(set(actual_state), set(reference_state))
            for name in actual_state:
                for field in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(actual_state[name][field],
                                               reference_state[name][field],
                                               rtol=1e-4, atol=1e-8)
        self.assertTrue(any(abs(float(actual_coefficients[0][role])
                                - float(actual_coefficients[1][role])) > 1e-9
                            for role in ROLES))

        saved = optimizer.state_dict()
        restored = engine._optimizer(model, job(spec))
        restored.load_state_dict(saved)
        loaded = restored.state_dict()
        self.assertEqual(loaded['param_groups'], saved['param_groups'])
        self.assertEqual(set(loaded['state']), set(saved['state']))
        for key in saved['state']:
            for field, value in saved['state'][key].items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(loaded['state'][key][field], value))

        shadow = ema.EMA(model, ema.RULE)
        self.assertEqual(set(shadow.state_dict()),
                         {'rule', 'model', 'update_count', 'copy_count', 'blend_count',
                          'last_update_success'})
        self.assertEqual(set(shadow.state_dict()['model']), set(model.state_dict()))
        self.assertEqual(set(optimizer.state_dict()['state']), set(saved['state']))


if __name__ == '__main__':
    unittest.main()
