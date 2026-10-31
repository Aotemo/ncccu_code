"""Synthetic temporary-public-head RNG contract for R129."""
import unittest

import torch
from torch import nn


class TinyPublic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 4)
        self.final_norm = nn.LayerNorm(4)
        self.public_head = nn.Linear(4, 1000)


def build(public_k=None):
    torch.manual_seed(20261029)
    model = TinyPublic()
    if public_k is not None:
        with torch.random.fork_rng(devices=[]):
            model.public_head = nn.Linear(4, public_k)
            accepted = {key: value.clone() for key, value in model.state_dict().items()}
            loaded = model.load_state_dict(accepted, strict=True)
            assert not loaded.missing_keys and not loaded.unexpected_keys
    model.public_head = nn.Linear(4, 1)
    age = nn.Parameter(torch.zeros(2))
    return model, age, torch.get_rng_state().clone()


class Small22KOnlyRngTest(unittest.TestCase):
    def test_temporary_complete_public_head_preserves_binary_age_and_rng(self):
        baseline, baseline_age, baseline_rng = build()
        changed, changed_age, changed_rng = build(public_k=7)
        self.assertTrue(torch.equal(baseline.public_head.weight, changed.public_head.weight))
        self.assertTrue(torch.equal(baseline.public_head.bias, changed.public_head.bias))
        self.assertTrue(torch.equal(baseline_age, changed_age))
        self.assertTrue(torch.equal(baseline_rng, changed_rng))
        self.assertTrue(torch.equal(baseline.final_norm.weight, changed.final_norm.weight))
        self.assertTrue(torch.equal(baseline.final_norm.bias, changed.final_norm.bias))
        self.assertEqual(tuple(changed.public_head.weight.shape), (1, 4))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
