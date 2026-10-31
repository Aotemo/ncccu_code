"""Positive CPU checks for the optional, unmerged Small pointwise adapter."""
import copy
import importlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime' / 'bundle'))
from retrain import age, config, engine, models
from retrain.persistence import capture_rng, restore_rng
from retrain.ema import EMA, RULE


def implementation(case):
    path = Path(models.__file__).with_name('low_rank.py')
    case.assertTrue(path.is_file(), 'optional pointwise low-rank implementation is absent')
    return importlib.import_module('retrain.low_rank')


def rng_equal(case, left, right):
    case.assertEqual(left['python'], right['python'])
    case.assertEqual(left['numpy'][0], right['numpy'][0])
    np.testing.assert_array_equal(left['numpy'][1], right['numpy'][1])
    case.assertEqual(left['numpy'][2:], right['numpy'][2:])
    case.assertTrue(torch.equal(left['torch'], right['torch']))
    case.assertEqual(left.keys(), right.keys())
    if left['cuda'] is None:
        case.assertIsNone(right['cuda'])
    else:
        case.assertTrue(torch.equal(left['cuda'], right['cuda']))


def toy_job(low_rank):
    return {'seed': 20261029, 'head_lr': 1e-3, 'backbone_lr': 1e-4,
            'weight_decay': 1e-4, 'head_lr_scope': 'last_linear',
            'age': {'enabled': True}, 'low_rank_policy': copy.deepcopy(low_rank.LOW_RANK_RULE)}


class Toy(nn.Module):
    def __init__(self, low_rank):
        super().__init__()
        self.network = nn.Module()
        self.network.features = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 4))
        self.network.classifier = nn.Sequential(nn.Linear(4, 1))
        self.age_weight = nn.Parameter(torch.zeros(2))
        for index in (0, 2):
            self.network.features[index] = low_rank.PointwiseLowRankLinear(
                self.network.features[index], 20261029, f'network.features.{index}')

    def forward(self, x, age_features):
        return self.network.classifier(self.network.features(x)).flatten() + age_features @ self.age_weight


class LowRankTest(unittest.TestCase):
    def test_zero_delta_rng_and_trainable_membership(self):
        low_rank = implementation(self)
        torch.set_num_threads(2)
        torch.manual_seed(83)
        random.seed(83)
        np.random.seed(83)
        original = nn.Linear(4, 6)
        original_names = {n: p.detach().clone() for n, p in original.named_parameters()}
        before = capture_rng()
        linear = low_rank.PointwiseLowRankLinear(original, 20261029, 'small.first')
        rng_equal(self, before, capture_rng())
        for name, value in original_names.items():
            self.assertTrue(torch.equal(linear.state_dict()[name], value))
        self.assertIs(linear.weight, original.weight)
        self.assertFalse(linear.weight.requires_grad)
        self.assertTrue(bool(torch.count_nonzero(linear.lora_A)))
        self.assertEqual(int(torch.count_nonzero(linear.lora_B)), 0)
        again = low_rank.PointwiseLowRankLinear(copy.deepcopy(original), 20261029, 'small.first')
        self.assertTrue(torch.equal(again.lora_A, linear.lora_A))
        x = torch.randn(3, 4, requires_grad=True)
        self.assertTrue(torch.equal(linear(x), F.linear(x, original.weight, original.bias)))
        linear(x).square().sum().backward()
        self.assertTrue(torch.equal(linear.lora_A.grad, torch.zeros_like(linear.lora_A)))
        self.assertIsNone(linear.weight.grad)
        linear.zero_grad(set_to_none=True)
        with torch.no_grad():
            linear.lora_B.copy_(torch.linspace(-.2, .3, linear.lora_B.numel()).reshape_as(linear.lora_B))
        a = linear.lora_A.detach().clone().requires_grad_()
        b = linear.lora_B.detach().clone().requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        reference = F.linear(reference_x, linear.weight, linear.bias) + F.linear(F.linear(reference_x, a), b)
        actual = linear(x)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        actual.square().sum().backward()
        reference.square().sum().backward()
        for actual_grad, expected in ((linear.lora_A.grad, a.grad), (linear.lora_B.grad, b.grad)):
            torch.testing.assert_close(actual_grad, expected, rtol=0, atol=0)
        # x also received the zero-delta backward; compare a fresh input gradient.
        x.grad = None
        linear(x).square().sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
        state = copy.deepcopy(linear.state_dict())
        linear.eval()
        eval_output = linear(x.detach())
        linear.train()
        self.assertTrue(torch.equal(eval_output, linear(x.detach())))
        for name, value in state.items():
            self.assertTrue(torch.equal(value, linear.state_dict()[name]))

        # The native structure check does not load public/training weights or forward images.
        job = config.default_job('lowrank-technical', 'convnext_small')
        job.update(precision='fp32', head_lr_scope='last_linear', seed=20261029,
                   backbone_lr=1e-4, head_lr=1e-3)
        job['age'] = age.make_spec({'rows': [{'name': 'a', 'sha256': 'a', 'age': 40},
                                            {'name': 'b', 'sha256': 'b', 'age': 60}],
                                   'fit_indices': [0, 1]})
        base = models.create_model(job, initialize=False)
        shared = {n: p.detach().clone() for n, p in base.named_parameters()}
        self.assertEqual(len(shared), 345)
        legacy = copy.deepcopy(config.job_contract(job))
        before = capture_rng()
        self.assertIs(low_rank.attach_low_rank(base, job), base)
        self.assertIsNone(low_rank.low_rank_proof(base, job))
        rng_equal(self, before, capture_rng())
        self.assertEqual(config.job_contract({**job, 'low_rank_policy': None}), legacy)
        job['low_rank_policy'] = copy.deepcopy(low_rank.LOW_RANK_RULE)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps({'jobs': [job]}))
            loaded_job = config.load_config(path, environ={})['jobs'][0]
            self.assertEqual(config.job_contract(loaded_job), config.job_contract(job))
        low_rank.attach_low_rank(base, job)
        rng_equal(self, before, capture_rng())
        for name, value in shared.items():
            self.assertTrue(torch.equal(base.state_dict()[name], value), name)
        optimizer = engine._optimizer(base, job)
        proof = low_rank.low_rank_proof(base, job, optimizer)
        self.assertEqual((len(proof['all_parameter_names']), len(proof['frozen_parameter_names']),
                          len(proof['trainable_parameter_names'])), (489, 342, 147))
        self.assertEqual((proof['all_parameter_elements'], proof['frozen_parameter_elements'],
                          proof['trainable_parameter_elements']), (50538339, 49454688, 1083651))
        self.assertEqual(len(proof['targets']), 72)
        self.assertEqual(optimizer.state, {})
        names = [n for g in optimizer.param_groups for n in g['param_names']]
        self.assertEqual(sorted(names), proof['trainable_parameter_names'])
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(g['initial_lr'] == (1e-4 if g['role'] == 'backbone' else 1e-3)
                            for g in optimizer.param_groups))
        self.assertEqual(config.job_contract(job)['low_rank_policy'], low_rank.LOW_RANK_RULE)
        self.assertEqual(models.feature_freeze_proof(base, job, optimizer), proof)
        self.assertEqual(low_rank.describe_low_rank(EMA(base, RULE).model, job),
                         low_rank.low_rank_proof(base, job))
        payload = {'job': job, 'identity': {'job': config.job_contract(job)},
                   'model': base.state_dict(), 'parameter_trainability': proof,
                   'optimizer': optimizer.state_dict(), 'successful_updates': 0}
        low_rank.validate_checkpoint(payload, job)
        low_rank.restore_base_reference(base, job, payload)
        self.assertEqual(low_rank.optimizer_update_proof(base, optimizer, 0)['observed_Adam_steps'], [])
        print('LOW_RANK_NATIVE: 345 shared exact; 72 targets; 489 total / 342 frozen / 147 trainable; Adam empty')


if __name__ == '__main__':
    unittest.main()
