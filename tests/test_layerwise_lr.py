"""Two synthetic checks for the R130 layerwise LR groups; A runs them on the Pod."""
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime' / 'bundle'))
from retrain import engine, layerwise_lr


class Holder(nn.Module):
    def __init__(self, shape=(2, 2)):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(*shape))


class NamedSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Module()
        features = nn.ModuleList([
            nn.Sequential(Holder(), Holder()),
            nn.Sequential(*[Holder() for _ in range(3)]),
            Holder(),
            nn.Sequential(*[Holder() for _ in range(3)]),
            Holder(),
            nn.Sequential(*[Holder((2, 1, 1)) if index == 3 else Holder() for index in range(27)]),
            Holder(),
            Holder()])
        self.network.features = features
        self.network.classifier = nn.Sequential(nn.LayerNorm(2), nn.Flatten(), nn.Dropout(),
                                                nn.Linear(2, 1))
        self.age_weight = nn.Parameter(torch.zeros(2))


def job():
    return {'head_lr': 1e-3, 'backbone_lr': 1e-4, 'weight_decay': 1e-4,
            'head_lr_scope': 'last_linear', 'age': {'enabled': True}}


def by_name(groups, name):
    return next(group for group in groups if name in group['param_names'])


def factor(successful, total=5, warmup=2):
    if warmup and successful < warmup:
        return (successful + 1) / warmup
    remaining = max(1, total - warmup - 1)
    progress = min(1.0, max(0.0, (successful - warmup) / remaining))
    return .01 + .99 * (1 + math.cos(math.pi * progress)) / 2


class LayerwiseLrTest(unittest.TestCase):
    def test_named_mapping_unique_coverage_wd_and_none_identity(self):
        model = NamedSmall()
        base = engine._optimizer(model, job()).param_groups
        snapshot = [{key: value for key, value in group.items() if key != 'params'}
                    for group in base]
        base_ids = [[id(parameter) for parameter in group['params']] for group in base]
        values = {id(p): p.detach().clone() for _, p in model.named_parameters()}
        rng = torch.random.get_rng_state()

        class NoScan:
            def named_parameters(self):
                raise AssertionError('None policy must not scan the model')

        self.assertIs(layerwise_lr.build_layerwise_groups(NoScan(), base, None), base)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        groups = layerwise_lr.build_layerwise_groups(model, base, layerwise_lr.LAYERWISE_LR_RULE)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for _, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, values[id(parameter)]))
        self.assertEqual([{key: value for key, value in group.items() if key != 'params'}
                          for group in base], snapshot)
        self.assertEqual([[id(parameter) for parameter in group['params']] for group in base],
                         base_ids)
        ids = [id(parameter) for group in groups for parameter in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(parameter) for _, parameter in model.named_parameters()})

        decay = layerwise_lr.LAYERWISE_LR_RULE['layer_decay']
        cases = [('network.features.0.0.weight', 0, decay ** 13, 1e-4, 1e-4),
                 ('network.features.1.0.weight', 1, decay ** 12, 1e-4, 1e-4),
                 ('network.features.2.weight', 2, decay ** 11, 1e-4, 1e-4),
                 ('network.features.3.0.weight', 2, decay ** 11, 1e-4, 1e-4),
                 ('network.features.4.weight', 3, decay ** 10, 1e-4, 1e-4),
                 ('network.features.5.0.weight', 3, decay ** 10, 1e-4, 1e-4),
                 ('network.features.5.2.weight', 3, decay ** 10, 1e-4, 1e-4),
                 ('network.features.5.3.weight', 4, decay ** 9, 1e-4, 1e-4),
                 ('network.features.5.26.weight', 11, decay ** 2, 1e-4, 1e-4),
                 ('network.features.6.weight', 12, decay, 1e-4, 1e-4),
                 ('network.features.7.weight', 12, decay, 1e-4, 1e-4),
                 ('network.classifier.0.weight', 13, 1., 1e-4, 0.),
                 ('network.classifier.3.weight', 13, 1., 1e-3, 1e-4),
                 ('network.classifier.3.bias', 13, 1., 1e-3, 0.),
                 ('age_weight', 13, 1., 1e-3, 1e-4)]
        for name, layer_id, scale, base_lr, weight_decay in cases:
            group = by_name(groups, name)
            self.assertEqual((group['layer_id'], group['lr_scale'], group['base_initial_lr'],
                              group['initial_lr'], group['weight_decay']),
                             (layer_id, scale, base_lr, base_lr * scale, weight_decay))
            target = dict(model.named_parameters())[name]
            position = next(index for index, parameter in enumerate(group['params'])
                            if parameter is target)
            self.assertEqual(group['param_names'][position], name)
        self.assertEqual(by_name(groups, 'network.features.5.3.weight')['layer_id'], 4)
        self.assertEqual(by_name(groups, 'network.classifier.0.weight')['role'], 'backbone')
        self.assertTrue(all(name.startswith('network.') or name == 'age_weight'
                            for group in groups for name in group['param_names']))
        identity = layerwise_lr.group_identity(model, groups, layerwise_lr.LAYERWISE_LR_RULE)
        self.assertEqual(len(identity), len(groups))
        self.assertTrue(all(set(row) == {'role', 'layer_id', 'lr_scale', 'base_initial_lr',
                                          'initial_lr', 'weight_decay', 'param_names'}
                            for row in identity))
        self.assertIsNone(layerwise_lr.group_identity(model, groups, None))

    def test_adamw_five_step_reference_with_scaled_initial_lr(self):
        model = NamedSmall()
        base = engine._optimizer(model, job()).param_groups
        groups = layerwise_lr.build_layerwise_groups(model, base, layerwise_lr.LAYERWISE_LR_RULE)
        optimizer = torch.optim.AdamW(groups)
        names = dict(model.named_parameters())
        owner = {id(p): group for group in groups for p in group['params']}
        reference = {name: p.detach().clone() for name, p in names.items()}
        first = {name: torch.zeros_like(p) for name, p in names.items()}
        second = {name: torch.zeros_like(p) for name, p in names.items()}
        betas = (.9, .999)
        eps = 1e-8
        for step in range(5):
            for index, (name, parameter) in enumerate(names.items()):
                parameter.grad = torch.full_like(parameter, .25 + (index % 3) * .1)
            engine._apply_lr(optimizer, step, 5, 2)
            for group in groups:
                self.assertAlmostEqual(group['lr'], group['initial_lr'] * factor(step), delta=1e-15)
            optimizer.step()
            for name, parameter in names.items():
                group = owner[id(parameter)]
                gradient = parameter.grad.detach()
                learning_rate = group['lr']
                decay = group['weight_decay']
                if decay:
                    reference[name].mul_(1 - learning_rate * decay)
                first[name].mul_(betas[0]).add_(gradient, alpha=1 - betas[0])
                second[name].mul_(betas[1]).addcmul_(gradient, gradient, value=1 - betas[1])
                count = step + 1
                first_hat = first[name] / (1 - betas[0] ** count)
                second_hat = second[name] / (1 - betas[1] ** count)
                reference[name].addcdiv_(first_hat, second_hat.sqrt().add_(eps),
                                         value=-learning_rate)
                torch.testing.assert_close(parameter, reference[name], rtol=1e-5, atol=1e-7)
                state = optimizer.state[parameter]
                torch.testing.assert_close(state['exp_avg'], first[name], rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(state['exp_avg_sq'], second[name], rtol=1e-5, atol=1e-7)
                self.assertEqual(int(state['step']), count)
        self.assertTrue(all(group['base_initial_lr'] == (1e-3 if group['role'] in ('head', 'age')
                                                         else 1e-4) for group in groups))


if __name__ == '__main__':
    unittest.main()
