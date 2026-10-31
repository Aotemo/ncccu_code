"""Two positive CPU risks: reflected filtering and reconstruction of a stateless graph."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import age, config, models
from retrain.persistence import CheckpointStore, capture_rng, canonical_digest, restore_rng

POLICY = {'version': 'convnext-small-tri3-downsample-v1', 'targets': [2, 4, 6],
          'convolution': 'native2-stride1-pad0', 'filter': 'reflect1-box2s1-box2s2'}


def implementation(test):
    test.assertIsNotNone(importlib.util.find_spec('retrain.tri3_downsample'),
                         'fixed Tri3 attachment and descriptor are missing')
    return importlib.import_module('retrain.tri3_downsample')


def technical_job():
    job = config.default_job('tri3-cpu-fixture', 'convnext_small')
    job.update(precision='fp32', head_lr_scope='last_linear', seed=20261029,
               image_size=512, prepared_size=512, physical_batch=8)
    job['age'] = age.make_spec({'rows': [{'name': 'a', 'sha256': 'a', 'age': 40},
                                        {'name': 'b', 'sha256': 'b', 'age': 60}],
                               'fit_indices': [0, 1]})
    return job


def rng_equal(test, before, after):
    test.assertEqual(before['python'], after['python'])
    test.assertEqual(before['numpy'][0], after['numpy'][0])
    test.assertTrue(np.array_equal(before['numpy'][1], after['numpy'][1]))
    test.assertEqual(before['numpy'][2:], after['numpy'][2:])
    test.assertTrue(torch.equal(before['torch'], after['torch']))
    test.assertIsNone(after['cuda'])


def toy_factory(job, initialize=False):
    """Only three real transition shapes, a head and age; no image backbone forward."""
    from torchvision.models.convnext import LayerNorm2d

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Module()
            self.network.features = nn.ModuleList([nn.Identity() for _ in range(8)])
            for index, inputs, outputs in ((2, 96, 192), (4, 192, 384), (6, 384, 768)):
                self.network.features[index] = nn.Sequential(LayerNorm2d(inputs, eps=1e-6),
                    nn.Conv2d(inputs, outputs, 2, stride=2))
            self.network.classifier = nn.Sequential(nn.LayerNorm(768, eps=1e-6),
                                                    nn.Dropout(.2), nn.Linear(768, 1))
            age.attach_age(self, job['age'])

    model = Toy()
    if job.get('tri3_downsample_policy') is not None:
        importlib.import_module('retrain.tri3_downsample').attach_tri3(model, job)
    return model


class Tri3DownsampleTest(unittest.TestCase):
    def test_filter_reflection_value_gradient_and_native_parameters_shapes(self):
        tri3 = implementation(self)
        job = technical_job()
        native = models.create_model(job, initialize=False)
        names = dict(native.named_parameters())
        saved_values = {name: parameter.detach().clone() for name, parameter in names.items()}
        targets = {index: tuple(native.network.features[index].children()) for index in (2, 4, 6)}
        before = capture_rng()
        job['tri3_downsample_policy'] = copy.deepcopy(POLICY)
        tri3.attach_tri3(native, job)
        rng_equal(self, before, capture_rng())
        self.assertEqual(set(native.state_dict()), set(saved_values))
        self.assertEqual(len(names), 345)
        self.assertEqual(dict(native.named_buffers()), {})
        for name, parameter in native.named_parameters():
            self.assertIs(parameter, names[name])
            self.assertTrue(parameter.requires_grad)
            self.assertTrue(torch.equal(parameter, saved_values[name]), name)
        description = tri3.describe_tri3(native, job)
        self.assertEqual(description['policy'], POLICY)
        self.assertEqual([row['feature_index'] for row in description['targets']], [2, 4, 6])
        for index, size, output_size in ((2, 127, 64), (4, 63, 32), (6, 31, 16)):
            stage = native.network.features[index]
            self.assertIs(stage[0], targets[index][0])
            self.assertIs(stage[1], targets[index][1])
            self.assertEqual(stage[1].stride, (1, 1))
            self.assertEqual(stage[1].padding, (0, 0))
            self.assertEqual([type(module) for module in stage[2:]],
                             [nn.ReflectionPad2d, nn.AvgPool2d, nn.AvgPool2d])
            tail = nn.Sequential(*list(stage.children())[2:])
            self.assertEqual(list(tail.parameters()), [])
            self.assertEqual(list(tail.buffers()), [])
            self.assertEqual(tuple(tail(torch.zeros(1, 1, size, size)).shape),
                             (1, 1, output_size, output_size))

        # Independent scalar binomial windows with explicit reflection, not two
        # pooling calls copied from production. Probe weights exercise both edges.
        height, width = 5, 7
        values = [[(r * width + c) / 9 + (r == 0) - 2 * (c == width - 1)
                   for c in range(width)] for r in range(height)]
        expected, expected_gradient = [], [[0.] * width for _ in range(height)]
        probe = [[(r + 1) * (c + 2) / 7 for c in range(4)] for r in range(3)]

        def reflect(index, length):
            return -index if index < 0 else 2 * length - 2 - index if index >= length else index

        for out_r, center_r in enumerate(range(0, height, 2)):
            row = []
            for out_c, center_c in enumerate(range(0, width, 2)):
                value = 0.
                for delta_r, weight_r in ((-1, 1), (0, 2), (1, 1)):
                    for delta_c, weight_c in ((-1, 1), (0, 2), (1, 1)):
                        r, c = reflect(center_r + delta_r, height), reflect(center_c + delta_c, width)
                        weight = weight_r * weight_c / 16
                        value += values[r][c] * weight
                        expected_gradient[r][c] += probe[out_r][out_c] * weight
                row.append(value)
            expected.append(row)
        x = torch.tensor(values, dtype=torch.float64).reshape(1, 1, height, width).requires_grad_()
        tail = nn.Sequential(*list(native.network.features[2].children())[2:])
        actual = tail(x)
        (actual * x.new_tensor(probe)).sum().backward()
        torch.testing.assert_close(actual, x.new_tensor(expected)[None, None], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(x.grad, x.new_tensor(expected_gradient)[None, None], rtol=1e-12, atol=1e-12)
        print(json.dumps({'tri3_native': {'parameter_tensors': len(names), 'buffers': 0,
                                         'descriptor': description, 'image_forward': False}}), flush=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_None_legacy_rng_identity_and_saved_inference_rebuild_executes_three_stages(self):
        tri3 = implementation(self)
        job = technical_job()
        before = capture_rng()
        legacy = models.create_model(job, initialize=False)
        legacy_exit = capture_rng()
        restore_rng(before)
        with_None = models.create_model({**job, 'tri3_downsample_policy': None}, initialize=False)
        rng_equal(self, legacy_exit, capture_rng())
        self.assertEqual(set(legacy.state_dict()), set(with_None.state_dict()))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, with_None.state_dict()[name]), name)
        for index in (2, 4, 6):
            self.assertEqual(len(with_None.network.features[index]), 2)
            self.assertEqual(with_None.network.features[index][1].stride, (2, 2))
        original_contract = config.job_contract(job)
        self.assertNotIn('tri3_downsample_policy', original_contract)
        self.assertEqual(config.job_contract({**job, 'tri3_downsample_policy': None}), original_contract)
        del legacy, with_None
        with tempfile.TemporaryDirectory(prefix='tri3-state-', dir=Path(__file__).parent) as folder:
            folder = Path(folder)
            path = folder / 'config.json'
            path.write_text(json.dumps({'jobs': [{**job, 'tri3_downsample_policy': None}]}))
            loaded_None = config.load_config(path, environ={})['jobs'][0]
            self.assertNotIn('tri3_downsample_policy', loaded_None)
            self.assertEqual(canonical_digest(config.job_contract(loaded_None)), canonical_digest(original_contract))
            path.write_text(json.dumps({'jobs': [{**job, 'tri3_downsample_policy': POLICY}]}))
            loaded_job = config.load_config(path, environ={})['jobs'][0]
            config.validate_job(loaded_job)
            model = toy_factory(loaded_job)
            identity = {'job': config.job_contract(loaded_job)}
            store = CheckpointStore(folder / 'checkpoints')
            store.save({'format_version': 1, 'mode': 'train', 'job': loaded_job, 'identity': identity,
                        'age': loaded_job['age'], 'manifest_digest': 'synthetic-only',
                        'model': model.state_dict(), 'successful_updates': 1})
            payload = store.load('last')
            self.assertEqual(payload['identity']['job']['tri3_downsample_policy'], POLICY)
            self.assertEqual(payload['job']['tri3_downsample_policy'], POLICY)
            from retrain.inference import _load_model
            rebuilt, restored_job, metadata, origin, selected_device = _load_model(
                folder / 'checkpoints', device='cpu', checkpoint='last',
                model_factory=toy_factory, variant='online')
            self.assertEqual(restored_job['tri3_downsample_policy'], POLICY)
            self.assertEqual(origin['tri3_downsample'], tri3.describe_tri3(rebuilt, restored_job))
            self.assertEqual(origin['identity']['job']['tri3_downsample_policy'], POLICY)
            self.assertEqual(str(selected_device), 'cpu')
            self.assertEqual(list(rebuilt.state_dict()), list(model.state_dict()))
            for index, channels in ((2, 96), (4, 192), (6, 384)):
                self.assertEqual([type(module) for module in rebuilt.network.features[index][2:]],
                                 [nn.ReflectionPad2d, nn.AvgPool2d, nn.AvgPool2d])
                self.assertEqual(rebuilt.network.features[index][1].stride, (1, 1))
                x = torch.arange(channels * 6 * 8, dtype=torch.float32).reshape(1, channels, 6, 8) / 137
                expected, actual = model.network.features[index](x), rebuilt.network.features[index](x)
                self.assertEqual(tuple(actual.shape[-2:]), (3, 4))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(json.dumps({'tri3_saved_inference': {'policy_retained': True,
                'three_stateless_stage_forwards_equal': True, 'gpu_initialized': torch.cuda.is_initialized()}}), flush=True)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
