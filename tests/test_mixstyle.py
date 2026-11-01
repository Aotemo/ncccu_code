"""Two CPU risks: detached local statistics/private RNG, and actual replay consumers."""
import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import age, config, models
from retrain.persistence import CheckpointStore, canonical_digest, capture_rng, restore_rng

POLICY = {'version': 'convnext-small-local8-mixstyle13-v1', 'sites': [1, 3],
          'probability': .5, 'alpha': .1, 'epsilon': 1e-6, 'variance_correction': 1,
          'scope': 'rank_local8', 'salt': 'mixstyle13-local8-v1',
          'rng': 'sha256-first8-big-private-python-v1'}


def implementation(test):
    test.assertIsNotNone(importlib.util.find_spec('retrain.mixstyle'),
                         'fixed local8 MixStyle implementation is missing')
    return importlib.import_module('retrain.mixstyle')


def technical_job():
    job = config.default_job('mixstyle-cpu-fixture', 'convnext_small')
    job.update(precision='fp32', seed=20261029, head_lr_scope='last_linear',
               image_size=512, prepared_size=512, physical_batch=8)
    job['age'] = age.make_spec({'rows': [{'name': 'a', 'sha256': 'a', 'age': 40},
                                        {'name': 'b', 'sha256': 'b', 'age': 60}],
                               'fit_indices': [0, 1]})
    return job


def rng_equal(test, left, right):
    test.assertEqual(left['python'], right['python'])
    test.assertEqual(left['numpy'][0], right['numpy'][0])
    test.assertTrue(np.array_equal(left['numpy'][1], right['numpy'][1]))
    test.assertEqual(left['numpy'][2:], right['numpy'][2:])
    test.assertTrue(torch.equal(left['torch'], right['torch']))
    test.assertIsNone(right['cuda'])


def scalar_reference(values, coefficients, permutation):
    """Independent Python scalar oracle: no torch mean/var or production helper."""
    batch, channels = len(values), len(values[0])
    height, width = len(values[0][0]), len(values[0][0][0])
    count = height * width
    means, sigmas = [], []
    for sample in values:
        mrow, srow = [], []
        for channel in sample:
            flat = [value for row in channel for value in row]
            mean = math.fsum(flat) / count
            variance = math.fsum((value - mean) ** 2 for value in flat) / (count - 1)
            mrow.append(mean); srow.append(math.sqrt(variance + 1e-6))
        means.append(mrow); sigmas.append(srow)
    output, diagonal = [], []
    for b in range(batch):
        lam, partner = coefficients[b], permutation[b]
        output.append([]); diagonal.append([])
        for c in range(channels):
            mixed_sigma = lam * sigmas[b][c] + (1-lam) * sigmas[partner][c]
            mixed_mean = lam * means[b][c] + (1-lam) * means[partner][c]
            output[b].append([[(value-means[b][c])/sigmas[b][c]*mixed_sigma+mixed_mean
                              for value in row] for row in values[b][c]])
            diagonal[b].append(mixed_sigma / sigmas[b][c])
    return output, diagonal


def oracle_draw(seed, context, site):
    """Contract oracle using the stdlib, never the production RNG wrapper."""
    key = ':'.join(map(str, (seed, 'mixstyle13-local8-v1',
                            context['attempted_before_increment'], context['rank'], context['micro'], site)))
    integer = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'big')
    private = random.Random(integer)
    gate = private.random()
    coefficients, permutation = None, None
    if gate <= .5:
        coefficients = [private.betavariate(.1, .1) for _ in range(8)]
        permutation = list(range(8)); private.shuffle(permutation)
    return integer, gate, coefficients, permutation


def toy_factory(job, initialize=False):
    """Keep real early CNBlocks, replace late stages: small saved CPU fixture only."""
    bare_job = {key: value for key, value in job.items() if key != 'mixstyle_policy'}
    model = models.create_model(bare_job, initialize=False)
    early = list(model.network.features.children())[:4]
    model.network.features = nn.Sequential(*early, nn.Conv2d(192, 768, 1),
                                           nn.Identity(), nn.Identity(), nn.Identity())
    if job.get('mixstyle_policy') is not None:
        importlib.import_module('retrain.mixstyle').attach_mixstyle(model, job)
    return model


class MixStyleTest(unittest.TestCase):
    def test_scalar_detached_gradient_and_private_attempt_rank_micro_site_rng(self):
        ms = implementation(self)
        values = [[[[4. if b == c == 0 else (b+1)*(c+2)+r*r/3-col/4
                     for col in range(3)] for r in range(2)] for c in range(2)] for b in range(8)]
        coefficients = [0., .125, .25, .5, .75, .875, 1., .375]
        permutation = [7, 6, 5, 4, 3, 2, 1, 0]
        expected, diagonal = scalar_reference(values, coefficients, permutation)
        x = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        probe = torch.arange(1, x.numel()+1, dtype=torch.float32).reshape_as(x) / 17
        actual = ms.mix_statistics(x, x.new_tensor(coefficients).reshape(8, 1, 1, 1),
                                   torch.tensor(permutation, dtype=torch.long))
        (actual * probe).sum().backward()
        torch.testing.assert_close(actual, x.new_tensor(expected), rtol=3e-6, atol=3e-5)
        torch.testing.assert_close(x.grad, x.new_tensor(diagonal)[:, :, None, None]*probe,
                                   rtol=4e-6, atol=3e-4)
        # Only output sample0 is probed: detached partner statistics give no
        # gradient to sample7, even with lambda0 and a zero-variance channel.
        separate = x.detach().clone().requires_grad_()
        ms.mix_statistics(separate, x.new_tensor(coefficients).reshape(8, 1, 1, 1),
                          torch.tensor(permutation))[0].sum().backward()
        self.assertTrue(torch.equal(separate.grad[1:], torch.zeros_like(separate.grad[1:])))
        torch.testing.assert_close(separate.grad[0], x.new_tensor(diagonal[0])[:, None, None].expand_as(x[0]),
                                   rtol=4e-6, atol=3e-4)
        before = capture_rng()
        self.assertIs(ms.apply(x, None, training=True, seed=20261029, context=None, site=1), x)
        self.assertIs(ms.apply(x, POLICY, training=False, seed=20261029, context=None, site=3), x)
        observed = []
        contexts = [{'attempted_before_increment': attempted, 'rank': rank, 'micro': micro}
                    for attempted, rank, micro in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))]
        for context in contexts:
            for site in (1, 3):
                trace = []
                result = ms.apply(x, POLICY, training=True, seed=20261029,
                                  context={**context, 'trace': trace}, site=site)
                integer, gate, lam, perm = oracle_draw(20261029, context, site)
                self.assertEqual(len(trace), 1)
                self.assertEqual(trace[0]['rng_seed_u64'], integer)
                self.assertEqual(trace[0]['gate_draw'], gate)
                self.assertEqual(trace[0]['active'], gate <= .5)
                self.assertEqual(trace[0]['lambdas'], lam)
                self.assertEqual(trace[0]['permutation'], perm)
                self.assertEqual({key: trace[0][key] for key in context}, context)
                self.assertEqual(trace[0]['site_index'], site)
                if lam is None:
                    self.assertIs(result, x)
                else:
                    expected_draw, _ = scalar_reference(values, lam, perm)
                    torch.testing.assert_close(result, x.new_tensor(expected_draw), rtol=3e-6, atol=3e-5)
                repeat = []
                torch.testing.assert_close(ms.apply(x, POLICY, training=True, seed=20261029,
                    context={**context, 'trace': repeat}, site=site), result, rtol=0, atol=0)
                self.assertEqual(repeat, trace)
                observed.append(trace[0])
        self.assertEqual(len({row['rng_seed_u64'] for row in observed}), len(observed))
        rng_equal(self, before, capture_rng())
        print(json.dumps({'mixstyle_formula_rng': {'traces': observed,
            'zero_variance_and_detached_partner_gradient': True, 'GPU_initialized': torch.cuda.is_initialized()}}))

    def test_native_None_contract_and_saved_inference_attempt_replay_eval_disabled(self):
        ms = implementation(self)
        job = technical_job()
        before = capture_rng()
        native = models.create_model(job, initialize=False)
        exit_native = capture_rng()
        restore_rng(before)
        native_None = models.create_model({**job, 'mixstyle_policy': None}, initialize=False)
        rng_equal(self, exit_native, capture_rng())
        parameters = dict(native.named_parameters())
        values = {name: value.detach().clone() for name, value in parameters.items()}
        network, targets = native.network, [native.network.features[i] for i in (1, 3)]
        for name, value in native_None.state_dict().items():
            self.assertTrue(torch.equal(value, values[name]), name)
        before_attach = capture_rng()
        ms.attach_mixstyle(native, {**job, 'mixstyle_policy': POLICY})
        rng_equal(self, before_attach, capture_rng())
        self.assertIs(native.network, network)
        self.assertEqual(len(parameters), 345)
        self.assertEqual(dict(native.named_buffers()), {})
        for name, value in native.named_parameters():
            self.assertIs(value, parameters[name]); self.assertTrue(torch.equal(value, values[name]))
        for i, target in zip((1, 3), targets):
            self.assertIs(native.network.features[i], target)
        native_description = ms.describe_mixstyle(native, {**job, 'mixstyle_policy': POLICY})
        self.assertEqual(native_description['policy'], POLICY)
        self.assertEqual([row['feature_index'] for row in native_description['sites']], [1, 3])
        self.assertEqual([row['channels'] for row in native_description['sites']], [96, 192])
        self.assertFalse(any(module._forward_hooks for module in native.modules()))
        del native, native_None, parameters, values, network, targets

        original_contract = config.job_contract(job)
        self.assertNotIn('mixstyle_policy', original_contract)
        self.assertEqual(config.job_contract({**job, 'mixstyle_policy': None}), original_contract)
        with tempfile.TemporaryDirectory(prefix='mixstyle-state-', dir=Path(__file__).parent) as folder:
            folder = Path(folder); path = folder/'config.json'
            path.write_text(json.dumps({'jobs': [{**job, 'mixstyle_policy': None}]}))
            loaded_None = config.load_config(path, environ={})['jobs'][0]
            self.assertNotIn('mixstyle_policy', loaded_None)
            self.assertEqual(canonical_digest(config.job_contract(loaded_None)), canonical_digest(original_contract))
            # Small real BinaryModel factories exercise the old training graph,
            # not a full native image forward. Existing SD/dropout RNG is replayed.
            plain, explicit_None = toy_factory(job), toy_factory(loaded_None)
            explicit_None.load_state_dict(plain.state_dict(), strict=True)
            image = torch.arange(8*3*32*32, dtype=torch.float32).reshape(8, 3, 32, 32) / 2048
            ages = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 11
            old_exit, old_value, old_gradient = None, None, None
            forward_rng = capture_rng()
            for index, (model, used_job) in enumerate(((plain, job), (explicit_None, loaded_None))):
                restore_rng(forward_rng)
                x = image.clone().requires_grad_(); model.train(); model.zero_grad(set_to_none=True)
                value = age.forward(model, x, ages, used_job)
                value.square().sum().backward()
                if index == 0:
                    old_value, old_gradient, old_exit = value.detach().clone(), x.grad.clone(), capture_rng()
                    old_parameter_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
                else:
                    torch.testing.assert_close(value, old_value, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, old_gradient, rtol=0, atol=0)
                    rng_equal(self, old_exit, capture_rng())
                    for name, p in model.named_parameters():
                        torch.testing.assert_close(p.grad, old_parameter_grads[name], rtol=0, atol=0)
            del plain, explicit_None

            path.write_text(json.dumps({'jobs': [{**job, 'mixstyle_policy': POLICY}]}))
            loaded_job = config.load_config(path, environ={})['jobs'][0]
            config.validate_job(loaded_job)
            model = toy_factory(loaded_job)
            self.assertEqual(ms.describe_mixstyle(model, loaded_job), native_description)
            # Select only a synthetic fixture key to exercise both sites. This
            # never constrains which real training attempts must activate.
            attempted = next(i for i in range(3, 67) if all(
                oracle_draw(job['seed'], {'attempted_before_increment': i, 'rank': 0, 'micro': 0}, site)[1] <= .5
                for site in (1, 3)))
            replay_rng = capture_rng()
            payload = {'format_version': 1, 'mode': 'train', 'job': loaded_job,
                'identity': {'job': config.job_contract(loaded_job)}, 'age': loaded_job['age'],
                'manifest_digest': 'synthetic-only', 'model': model.state_dict(),
                'successful_updates': attempted-2, 'attempted_updates': attempted, 'rng': replay_rng}
            store = CheckpointStore(folder/'checkpoints'); store.save(payload)
            loaded = store.load('last')
            self.assertEqual(loaded['identity']['job']['mixstyle_policy'], POLICY)
            self.assertEqual(loaded['job']['mixstyle_policy'], POLICY)
            from retrain.inference import _load_model
            rebuilt, restored_job, _, origin, device = _load_model(folder/'checkpoints', checkpoint='last',
                device='cpu', variant='online', model_factory=toy_factory)
            self.assertEqual(origin['mixstyle'], native_description)
            self.assertEqual(restored_job['mixstyle_policy'], POLICY)
            self.assertEqual(str(device), 'cpu')
            traces, outputs = [], []
            for current in (model, rebuilt):
                restore_rng(loaded['rng']); current.train()
                keys_before = set(vars(current)); trace = []
                context = {'attempted_before_increment': loaded['attempted_updates'],
                           'rank': 0, 'micro': 0, 'trace': trace}
                outputs.append(age.forward(current, image, ages, restored_job, mixstyle_context=context))
                self.assertEqual(set(vars(current)), keys_before)
                self.assertEqual([row['site_index'] for row in trace], [1, 3])
                self.assertTrue(all(row['active'] for row in trace))
                self.assertTrue(all(row['attempted_before_increment'] == attempted for row in trace))
                traces.append(trace)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            self.assertEqual(traces[0], traces[1])
            from retrain.ema import EMA, RULE
            ema = EMA(model, RULE)
            self.assertEqual(ms.describe_mixstyle(ema.model, restored_job), native_description)
            model.eval(); rebuilt.eval(); trace = []
            eval_rng = capture_rng()
            with torch.no_grad():
                expected_eval = age.residual(model, model.network(image).flatten(), ages)
                for current in (model, rebuilt, ema.model):
                    actual_eval = age.forward(current, image, ages, restored_job,
                        mixstyle_context={'attempted_before_increment': attempted, 'rank': 0,
                                          'micro': 0, 'trace': trace})
                    torch.testing.assert_close(actual_eval, expected_eval, rtol=0, atol=0)
            self.assertEqual(trace, [])
            rng_equal(self, eval_rng, capture_rng())
            self.assertFalse(any(module._forward_hooks for module in model.modules()))
            print(json.dumps({'mixstyle_saved_replay': {'native_parameter_tensors': 345,
                'native_buffers': 0, 'descriptor': native_description,
                'fixture_attempted_updates': attempted, 'fixture_successful_updates': attempted-2,
                'actual_replayed_traces': traces[1], 'None_value_gradient_rng_unchanged': True,
                'online_EMA_eval_disabled': True, 'GPU_initialized': torch.cuda.is_initialized()}}))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
