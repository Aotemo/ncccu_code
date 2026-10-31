"""Two toy checks for the R124 Small content-attention-pool branch.

Parent A runs these once on the guarded Pod.  The shell mirrors the production
network.layout (features / avgpool / classifier) but uses a tiny 16x16 input and
no real model, weights, images or cache.
"""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import attention_pool, ema, engine


def toy_small(seed):
    torch.manual_seed(seed)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Conv2d(3, 768, 3, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Flatten(1), nn.LayerNorm(768),
                                            nn.Dropout(.1), nn.Linear(768, 1))

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = Network()

    return Shell()


def job(model, attention=None):
    return {'model': model, 'precision': 'fp32', 'head_lr': .001, 'backbone_lr': 1e-4,
            'weight_decay': .0001, 'attention_pool': attention}


class ContentAttentionPoolSmallTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        self.images = torch.randn(2, 3, 16, 16)

    def test_none_and_zero_q_keep_gap_output_and_expose_q_gradient(self):
        plain = toy_small(5)
        attention_pool.attach(plain, job('convnext_small', None))
        self.assertFalse(hasattr(plain, 'attention_pool_weight'))
        self.assertEqual(tuple(plain.network.avgpool(plain.network.features(self.images)).shape),
                         (2, 768, 1, 1))

        baseline, target = toy_small(5), toy_small(5)
        state = torch.random.get_rng_state()
        attention_pool.attach(target, job('convnext_small', attention_pool.ATTENTION_POOL_RULE))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        weight = target.attention_pool_weight
        self.assertEqual(tuple(weight.shape), (1, 768))
        self.assertEqual(weight.dtype, torch.float32)
        self.assertTrue(bool((weight == 0).all()))
        baseline.eval()
        target.eval()
        self.assertEqual(set(target.state_dict()), set(baseline.state_dict()) | {'attention_pool_weight'})
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[name], value))
        with torch.no_grad():
            gap = baseline.network.classifier(baseline.network.avgpool(
                baseline.network.features(self.images))).flatten()
            pooled = attention_pool.image_logits(target, self.images)
        torch.testing.assert_close(pooled, gap, rtol=0, atol=0)

        attention_pool.image_logits(target, self.images).sum().backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(bool(torch.isfinite(weight.grad).all()))
        self.assertGreater(int(torch.count_nonzero(weight.grad)), 0)

        tiny = toy_small(6)
        attention_pool.attach(tiny, job('convnext_tiny', attention_pool.ATTENTION_POOL_RULE))
        self.assertEqual(tuple(tiny.attention_pool_weight.shape), (1, 768))

    def test_nonzero_q_matches_independent_oracle_optimizer_and_ema(self):
        torch.manual_seed(7)
        features = torch.randn(2, 768, 4, 4)
        q = torch.linspace(-.4, .4, 768).reshape(1, 768)
        model = toy_small(8)
        attention_pool.attach(model, job('convnext_small', attention_pool.ATTENTION_POOL_RULE))
        with torch.no_grad():
            model.attention_pool_weight.copy_(q)
        local = features.clone().requires_grad_()
        pooled = attention_pool.pool_features(model, local)
        upstream = torch.linspace(-1., 1., pooled.numel()).reshape_as(pooled)
        (pooled * upstream).sum().backward()
        actual_grad_features = local.grad.detach()
        actual_grad_q = model.attention_pool_weight.grad.detach()

        oracle_features = features.double().clone().requires_grad_()
        oracle_q = q.double().clone().requires_grad_()
        locations = oracle_features.flatten(2).transpose(1, 2)
        normalized = F.layer_norm(locations, (768,), eps=1e-6)
        weights = torch.softmax((normalized * oracle_q).sum(-1) / (768 ** .5), dim=1)
        centered = weights - (1.0 / locations.shape[1])
        correction = (oracle_features.flatten(2) * centered.unsqueeze(1)).sum(2)[:, :, None, None]
        reference = F.adaptive_avg_pool2d(oracle_features, 1) + correction
        (reference * upstream.double()).sum().backward()
        torch.testing.assert_close(pooled.double(), reference, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual_grad_features.double(), oracle_features.grad,
                                   rtol=2e-3, atol=1e-5)
        torch.testing.assert_close(actual_grad_q.double(), oracle_q.grad, rtol=2e-3, atol=1e-5)

        grouped_job = job('convnext_small', attention_pool.ATTENTION_POOL_RULE)
        flat_job = job('convnext_small', None)
        baseline = toy_small(8)
        names = {id(p): name for name, p in model.named_parameters()}
        baseline_names = {id(p): name for name, p in baseline.named_parameters()}
        optimizer = engine._optimizer(model, grouped_job)
        optimizer_baseline = engine._optimizer(baseline, flat_job)
        q_entries = [p for group in optimizer.param_groups for p in group['params']
                     if p is model.attention_pool_weight]
        self.assertEqual(len(q_entries), 1)
        q_group = next(group for group in optimizer.param_groups
                       if any(p is model.attention_pool_weight for p in group['params']))
        self.assertEqual((q_group['lr'], q_group['weight_decay'], q_group['role']),
                         (.001, .0001, 'head'))
        table = {names[id(p)]: (g['lr'], g['weight_decay'], g['role'])
                 for g in optimizer.param_groups for p in g['params']}
        baseline_table = {baseline_names[id(p)]: (g['lr'], g['weight_decay'], g['role'])
                          for g in optimizer_baseline.param_groups for p in g['params']}
        self.assertEqual(set(table) - set(baseline_table), {'attention_pool_weight'})
        self.assertEqual(set(baseline_table) - set(table), set())
        self.assertEqual({key: value for key, value in table.items() if key != 'attention_pool_weight'},
                         baseline_table)
        self.assertEqual(table['attention_pool_weight'], (.001, .0001, 'head'))
        self.assertEqual(table['network.classifier.1.weight'][1], 0.0)

        shadow_state = {'rule': copy.deepcopy(ema.RULE),
                        'model': {name: value.detach().clone()
                                  for name, value in model.state_dict().items()},
                        'update_count': 475, 'copy_count': 474, 'blend_count': 1,
                        'last_update_success': 475}
        shadow = ema.EMA(model, ema.RULE, state=shadow_state, successful=475)
        self.assertEqual(shadow.model.attention_pool_spec, attention_pool.ATTENTION_POOL_RULE)
        self.assertTrue(torch.equal(shadow.model.attention_pool_weight, q))
        with torch.no_grad():
            model.attention_pool_weight.add_(.25)
        current = model.attention_pool_weight.detach().clone()
        shadow.update(model, 476)
        expected = q.mul(ema.RULE['decay']).add(current, alpha=1 - ema.RULE['decay'])
        torch.testing.assert_close(shadow.model.attention_pool_weight, expected,
                                   rtol=1e-7, atol=1e-9)
        self.assertEqual(shadow.state_dict()['blend_count'], 2)
        self.assertIn('attention_pool_weight', shadow.state_dict()['model'])


if __name__ == '__main__':
    unittest.main()
