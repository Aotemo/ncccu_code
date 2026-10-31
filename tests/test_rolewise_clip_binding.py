"""R126 policy binding and mutation order, using only tiny CPU tensors."""
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime' / 'bundle'))
from retrain import engine
from retrain.gradient_clip import ROLEWISE_CLIP_RULE, build_role_groups


class RolewiseClipBindingTest(unittest.TestCase):
    def test_science_default_and_world_finite_before_mutation(self):
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        current = json.loads((ROOT / 'configs/R126-SCIENCE.json').read_text())
        self.assertEqual(current['candidate']['gradient_clip_policy'], ROLEWISE_CLIP_RULE)
        self.assertEqual(current['job']['gradient_clip_policy'], ROLEWISE_CLIP_RULE)
        self.assertEqual(current['candidate']['readout'], original['candidate']['readout'])
        self.assertEqual(current['initial_model_tensor_sha256'], original['initial_model_tensor_sha256'])
        for key in ('candidate_profile', 'official_evaluation', 'comparison_note'):
            current.pop(key, None)
            original.pop(key, None)
        for key in ('id', 'experiment_order', 'gradient_clip_policy'):
            current['candidate'].pop(key, None)
            original['candidate'].pop(key, None)
        current['job'].pop('gradient_clip_policy')
        self.assertEqual(current, original)

        parameters = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
        for parameter, vector in zip(parameters, ([3., 4.], [0., 2.], [0., .5])):
            parameter.grad = torch.tensor(vector)
        optimizer = torch.optim.AdamW([{'params': [parameter], 'role': role}
                                        for parameter, role in zip(parameters, ROLEWISE_CLIP_RULE['roles'])])
        self.assertIsNone(build_role_groups(optimizer, None))
        legacy = [parameter.grad.clone() for parameter in parameters]
        finite, norms, coefficients = engine._clip_gradients(parameters, None)
        self.assertTrue(finite)
        self.assertIsNone(norms)
        self.assertIsNone(coefficients)
        for parameter, original_gradient in zip(parameters, legacy):
            torch.testing.assert_close(parameter.grad, original_gradient / (29.25 ** .5 + 1e-6))

        for parameter, original_gradient in zip(parameters, legacy):
            parameter.grad = original_gradient.clone()
        groups = build_role_groups(optimizer, ROLEWISE_CLIP_RULE)
        calls = []
        before = [parameter.grad.clone() for parameter in parameters]
        finite, norms, coefficients = engine._clip_gradients(
            parameters, groups, lambda bad: calls.append(bad) or True)
        self.assertFalse(finite)
        self.assertIsNone(coefficients)
        self.assertEqual(calls, [False])
        for parameter, saved in zip(parameters, before):
            torch.testing.assert_close(parameter.grad, saved, rtol=0, atol=0)

        calls.clear()
        finite, norms, coefficients = engine._clip_gradients(
            parameters, groups, lambda bad: calls.append(bad) or False)
        self.assertTrue(finite)
        self.assertEqual(calls, [False])
        self.assertEqual(set(norms), set(ROLEWISE_CLIP_RULE['roles']))
        self.assertLess(coefficients['backbone'], 1.0)
        self.assertLess(coefficients['head'], 1.0)
        self.assertEqual(coefficients['age'], 1.0)
        for parameter, saved, role in zip(parameters, before, ROLEWISE_CLIP_RULE['roles']):
            torch.testing.assert_close(parameter.grad, saved * coefficients[role])


if __name__ == '__main__':
    unittest.main()
