"""Small CPU tensor checks for BCE compatibility and focal differentiation."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain.classification_loss import classification_loss, validate_loss


FOCAL = {'version': 'binary-focal-v1', 'kind': 'binary_focal', 'gamma': 2.0,
         'alpha': None, 'reduction': 'mean', 'dtype': 'float32',
         'modulation_detached': False}


class ClassificationLossTest(unittest.TestCase):
    def test_default_and_gamma_zero_match_BCE_values_and_gradients(self):
        import torch
        from torch.nn import functional as F

        for specification, dtype in ((None, torch.float32), (None, torch.float64),
                                     ({**FOCAL, 'gamma': 0.0}, torch.float32)):
            with self.subTest(specification=specification, dtype=dtype):
                logits = torch.tensor([-3., -.4, 0., .8, 2.5], dtype=dtype,
                                      requires_grad=True)
                reference_logits = logits.detach().clone().requires_grad_(True)
                targets = torch.tensor([0., 1., 1., 0., 1.], dtype=dtype)
                if specification is None:
                    self.assertIsNone(validate_loss(None))
                    actual = classification_loss(logits, targets)
                else:
                    actual = classification_loss(logits, targets, specification)
                expected = F.binary_cross_entropy_with_logits(reference_logits, targets)
                actual_grad, = torch.autograd.grad(actual, logits)
                expected_grad, = torch.autograd.grad(expected, reference_logits)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)

    def test_gamma_two_matches_probability_reference_and_non_detached_gradient(self):
        import torch

        logits = torch.tensor([-2.25, -.75, 0., 1.125, 2.5], dtype=torch.float64,
                              requires_grad=True)
        reference_logits = logits.detach().clone().requires_grad_(True)
        targets = torch.tensor([0., 1., 0., 1., 0.], dtype=torch.float64)
        actual = classification_loss(logits, targets, FOCAL)
        # Moderate inputs allow an independent probability/log reference, without
        # reusing the production BCE-with-logits implementation or exp(-BCE).
        probability = reference_logits.float().sigmoid()
        pt = torch.where(targets.bool(), probability, 1. - probability)
        modulation = (1. - pt).square()
        expected = (-pt.log() * modulation).mean()
        detached_reference = (-pt.log() * modulation.detach()).mean()
        actual_grad, = torch.autograd.grad(actual, logits)
        expected_grad, = torch.autograd.grad(expected, reference_logits, retain_graph=True)
        detached_grad, = torch.autograd.grad(detached_reference, reference_logits)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-6, atol=1e-7)
        self.assertGreater((actual_grad - detached_grad).abs().max().item(), 1e-3)

    def test_extreme_finite_logits_have_finite_loss_and_gradients(self):
        import torch

        logits = torch.tensor([-10000., 10000., -1000., 1000., 0.],
                              dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0., 1., 1., 0., 1.], dtype=torch.float32)
        loss = classification_loss(logits, targets, FOCAL)
        gradient, = torch.autograd.grad(loss, logits)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(gradient).all().item())


if __name__ == '__main__':
    unittest.main()
