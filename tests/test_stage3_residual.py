"""Two toy checks for the R125 stage3 GAP residual readout.

Parent A runs these once on the guarded Pod after wiring the helper and the
optimizer role.  No real Small model, public weights, images or assets are used.
"""
import copy
from contextlib import contextmanager
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import age, ema, engine, stage3_residual as helper


def toy_small(seed):
    torch.manual_seed(seed)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 3, padding=1), nn.GELU(),
                nn.Conv2d(96, 192, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(192, 384, 3, stride=2, padding=1), nn.Identity(),
                nn.Conv2d(384, 768, 3, stride=2, padding=1), nn.GELU())
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Flatten(1), nn.LayerNorm(768),
                                            nn.Dropout(.1), nn.Linear(768, 1))

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = Network()

    return Shell()


def job(residual):
    return {'model': 'convnext_small', 'precision': 'fp32', 'head_lr': .001,
            'backbone_lr': 1e-4, 'weight_decay': .0001, 'stage3_residual': residual}


@contextmanager
def count_forward(model):
    calls = {}
    handles = []

    def record(name):
        def hook(module, args, output):
            calls[name] = calls.get(name, 0) + 1
        return hook

    for index, module in enumerate(model.network.features):
        handles.append(module.register_forward_hook(record(('features', index))))
    handles.append(model.network.avgpool.register_forward_hook(record(('avgpool',))))
    handles.append(model.network.classifier.register_forward_hook(record(('classifier',))))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


class Stage3ResidualTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        self.images = torch.randn(2, 3, 16, 16)

    def test_none_compatibility_single_pass_q0_identity_and_graph(self):
        plain = toy_small(5)
        helper.attach(plain, job(None))
        self.assertFalse(hasattr(plain, 'stage3_residual_weight'))
        old = plain.network.classifier(plain.network.avgpool(plain.network.features(self.images))).flatten()
        self.assertEqual(tuple(old.shape), (2,))

        baseline, target = toy_small(6), toy_small(6)
        state = torch.random.get_rng_state()
        helper.attach(target, job(helper.STAGE3_RESIDUAL_RULE))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        weight = target.stage3_residual_weight
        self.assertEqual(tuple(weight.shape), (1, 384))
        self.assertEqual(weight.dtype, torch.float32)
        self.assertTrue(bool((weight == 0).all()))
        self.assertEqual(set(target.state_dict()), set(baseline.state_dict()) | {'stage3_residual_weight'})
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[name], value))

        baseline.train()
        target.train()
        torch.manual_seed(2026)
        expected_calls = {('features', index): 1 for index in range(8)}
        expected_calls.update({('avgpool',): 1, ('classifier',): 1})
        with patch.object(age, 'residual',
                          side_effect=AssertionError('helper must not re-enter age')) as age_call, \
                count_forward(target) as calls:
            logits = helper.image_logits(target, self.images)
        age_call.assert_not_called()
        self.assertEqual(calls, expected_calls)
        self.assertTrue(target.training)
        self.assertTrue(target.network.classifier[2].training)
        rng_after = torch.random.get_rng_state()
        torch.manual_seed(2026)
        reference = baseline.network.classifier(baseline.network.avgpool(
            baseline.network.features(self.images))).flatten()
        self.assertTrue(torch.equal(rng_after, torch.random.get_rng_state()))
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)

        upstream = torch.linspace(-1., 1., logits.numel())
        (logits * upstream).sum().backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(bool(torch.isfinite(weight.grad).all()))
        self.assertGreater(int(torch.count_nonzero(weight.grad)), 0)
        for module, name in ((target.network.classifier[-1], 'linear'),
                             (target.network.classifier[1], 'final_norm'),
                             (target.network.features[4], 'stage3_conv')):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(gradients and all(g is not None and bool(torch.isfinite(g).all())
                                              for g in gradients), name)

    def test_nonzero_q_matches_double_oracle_optimizer_and_ema(self):
        torch.manual_seed(21)
        images = torch.randn(2, 3, 16, 16)
        q = torch.linspace(-.3, .3, 384).reshape(1, 384)
        model = toy_small(8)
        helper.attach(model, job(helper.STAGE3_RESIDUAL_RULE))
        with torch.no_grad():
            model.stage3_residual_weight.copy_(q)
        model.eval()
        local = images.clone().requires_grad_()
        logits = helper.image_logits(model, local)
        self.assertFalse(model.training)
        upstream = torch.linspace(-1., 1., logits.numel())
        (logits * upstream).sum().backward()
        actual_grad_images = local.grad.detach()
        actual_grad_q = model.stage3_residual_weight.grad.detach()

        reference = copy.deepcopy(model).double()
        ref_images = images.double().clone().requires_grad_()
        hidden = ref_images
        for layer in reference.network.features[:6]:
            hidden = layer(hidden)
        stage3 = hidden
        for layer in reference.network.features[6:]:
            hidden = layer(hidden)
        logit_image = reference.network.classifier(reference.network.avgpool(hidden)).flatten()
        residual = F.layer_norm(F.adaptive_avg_pool2d(stage3, 1).flatten(1), (384,),
                                weight=None, bias=None, eps=1e-6)
        expected = logit_image + (residual * reference.stage3_residual_weight).sum(dim=1)
        (expected * upstream.double()).sum().backward()
        torch.testing.assert_close(logits.double(), expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual_grad_images.double(), ref_images.grad,
                                   rtol=2e-3, atol=1e-5)
        torch.testing.assert_close(actual_grad_q.double(), reference.stage3_residual_weight.grad,
                                   rtol=2e-3, atol=1e-5)

        baseline = toy_small(8)
        optimizer = engine._optimizer(model, job(helper.STAGE3_RESIDUAL_RULE))
        optimizer_baseline = engine._optimizer(baseline, job(None))
        q_entries = [p for group in optimizer.param_groups for p in group['params']
                     if p is model.stage3_residual_weight]
        self.assertEqual(len(q_entries), 1)
        q_group = next(group for group in optimizer.param_groups
                       if any(p is model.stage3_residual_weight for p in group['params']))
        self.assertEqual((q_group['lr'], q_group['weight_decay'], q_group['role']),
                         (.001, .0001, 'head'))
        names = {id(p): name for name, p in model.named_parameters()}
        baseline_names = {id(p): name for name, p in baseline.named_parameters()}
        table = {names[id(p)]: (g['lr'], g['weight_decay'], g['role'])
                 for g in optimizer.param_groups for p in g['params']}
        baseline_table = {baseline_names[id(p)]: (g['lr'], g['weight_decay'], g['role'])
                          for g in optimizer_baseline.param_groups for p in g['params']}
        self.assertEqual(set(table) - set(baseline_table), {'stage3_residual_weight'})
        self.assertEqual(set(baseline_table) - set(table), set())
        self.assertEqual({key: value for key, value in table.items()
                          if key != 'stage3_residual_weight'}, baseline_table)
        self.assertEqual(table['stage3_residual_weight'], (.001, .0001, 'head'))
        self.assertEqual(table['network.classifier.1.weight'][1], 0.0)
        self.assertEqual(optimizer.state_dict()['state'], {})

        shadow_state = {'rule': copy.deepcopy(ema.RULE),
                        'model': {name: value.detach().clone()
                                  for name, value in model.state_dict().items()},
                        'update_count': 475, 'copy_count': 474, 'blend_count': 1,
                        'last_update_success': 475}
        shadow = ema.EMA(model, ema.RULE, state=shadow_state, successful=475)
        self.assertEqual(shadow.model.stage3_residual_spec, helper.STAGE3_RESIDUAL_RULE)
        self.assertTrue(torch.equal(shadow.model.stage3_residual_weight, q))
        with torch.no_grad():
            model.stage3_residual_weight.add_(.25)
        current = model.stage3_residual_weight.detach().clone()
        shadow.update(model, 476)
        expected_q = q.mul(ema.RULE['decay']).add(current, alpha=1 - ema.RULE['decay'])
        torch.testing.assert_close(shadow.model.stage3_residual_weight, expected_q,
                                   rtol=1e-7, atol=1e-9)
        self.assertEqual(shadow.state_dict()['blend_count'], 2)
        self.assertIn('stage3_residual_weight', shadow.state_dict()['model'])


if __name__ == '__main__':
    unittest.main()
