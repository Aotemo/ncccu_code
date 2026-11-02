"""Fixed fit-only gamma pixels over the unchanged sample-key geometry."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import config, data

POLICY = 'gamma-reciprocal075-v1'
BaseRandom = random.Random


def original_canvas(image, size, seed, training=True, identity=False):
    from PIL import Image, ImageEnhance, ImageOps
    scale, dx, dy = 1., .5, .5
    if training:
        rng = BaseRandom(seed)
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            factor = rng.uniform(.9, 1.1)
            if not identity:
                image = enhancer(image).enhance(factor)
        image = image.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR,
                             expand=True, fillcolor=(0, 0, 0))
        scale, dx, dy = rng.uniform(.85, 1.), rng.uniform(.25, .75), rng.uniform(.25, .75)
    ratio = size * scale / max(image.size)
    target = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
    image = image.resize(target, Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size, size))
    canvas.paste(image, (round((size - target[0]) * dx), round((size - target[1]) * dy)))
    return canvas


def expected_gamma(seed):
    key = int.from_bytes(hashlib.sha256(
        f'{seed}:training-gamma-reciprocal075-v1'.encode()).digest()[:8], 'big')
    return BaseRandom(key).choice((.75, 1., 4/3)), key


def tensor_oracle(canvas, gamma=1.):
    import numpy as np
    import torch
    # Independent per-pixel FP64 calculation, rather than the implementation LUT.
    pixels = np.rint(255 * np.power(np.asarray(canvas).astype(np.float64) / 255, gamma)).astype(np.uint8)
    tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(255)
    return tensor.sub_(tensor.new_tensor((.485, .456, .406))[:, None, None]).div_(
        tensor.new_tensor((.229, .224, .225))[:, None, None])


class GammaPolicyTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        y, x = np.indices((47, 61))
        self.image = Image.fromarray(np.stack((x*7 % 256, y*11 % 256, (x*3+y*5) % 256), -1).astype(np.uint8))
        self.rows = [dict(name=f'synthetic-{i}.png', relative_path=f'train/synthetic-{i}.png',
                         sha256='a'*64, label=int(i == 0), age=str(30+i)) for i in range(8)]

    def make(self, policy=POLICY, training=True, size=48, age=None):
        with patch.object(data, 'configured_cache', return_value=None):
            try:
                return data.ImageDataset(self.root, self.rows, size, training,
                    training_scale=(.85, 1.), training_photometric=policy, age=age)
            except ValueError as error:
                self.fail(f'fixed gamma policy not accepted: {error}')

    def test_training_pixel_oracle_replay_and_independent_rng(self):
        import numpy as np
        import torch
        from PIL import Image
        dataset = self.make()
        before = random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()
        source_pixels = np.asarray(self.image).copy()
        observed = set()
        for seed in range(16):
            gamma, gamma_key = expected_gamma(seed)
            observed.add(gamma)
            canvas = original_canvas(self.image.copy(), 48, seed)
            seeds = []
            def factory(value):
                seeds.append(value)
                return BaseRandom(value)
            with patch.object(data.random, 'Random', side_effect=factory):
                actual = dataset._image_tensor(self.image.copy(), seed)
            self.assertEqual(seeds, [seed, gamma_key])
            self.assertTrue(torch.equal(actual, tensor_oracle(canvas, gamma)))
            self.assertTrue(torch.equal(actual, dataset._image_tensor(self.image.copy(), seed)))
        self.assertEqual(observed, {.75, 1., 4/3})
        # A full RGB byte ramp exercises the complete lookup, including 0/255.
        ramp = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (256, 1, 3)))
        class FixedGeometry:
            def random(self): return 1.
            def uniform(self, low, high):
                return 1. if (low, high) == (.85, 1.) else (low+high)/2
        for seed in range(16):
            gamma, gamma_key = expected_gamma(seed)
            with patch.object(data.random, 'Random', side_effect=lambda value:
                              BaseRandom(value) if value == gamma_key else FixedGeometry()):
                actual = self.make(size=256)._image_tensor(ramp.copy(), seed)
            self.assertTrue(torch.equal(actual, tensor_oracle(ramp, gamma)))
        self.assertTrue(np.array_equal(source_pixels, np.asarray(self.image)))
        self.assertEqual(before[0], random.getstate())
        self.assertEqual(before[1][0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(before[1][1], np.random.get_state()[1]))
        self.assertEqual(before[1][2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_None_identity_eval_and_actual_P8_512_schema(self):
        import numpy as np
        import torch
        from retrain import age
        for policy in (None, 'identity-v1'):
            actual = self.make(policy=policy)._image_tensor(self.image.copy(), 20261029)
            expected = tensor_oracle(original_canvas(self.image.copy(), 48, 20261029,
                                                      identity=policy == 'identity-v1'))
            self.assertTrue(torch.equal(actual, expected))
        expected = tensor_oracle(original_canvas(self.image.copy(), 48, 0, training=False))
        for policy in (None, 'identity-v1', POLICY):
            with patch.object(data.random, 'Random', side_effect=AssertionError('eval may not sample')):
                actual = self.make(policy=policy, training=False)._image_tensor(self.image.copy(), 99)
            self.assertTrue(torch.equal(actual, expected))
        spec = age.make_spec({'rows': self.rows, 'fit_indices': list(range(8))})
        dataset = self.make(size=512, age=spec)
        with patch.object(dataset, '_read_image', side_effect=lambda row: self.image.copy()):
            samples = [dataset[(i, 20261029+i)] for i in range(8)]
        batch = torch.stack([s[0] for s in samples])
        self.assertEqual(tuple(batch.shape), (8, 3, 512, 512))
        self.assertEqual(batch.dtype, torch.float32)
        self.assertTrue(torch.isfinite(batch).all())
        self.assertEqual([s[1] for s in samples], [float(r['label']) for r in self.rows])
        self.assertEqual([s[2] for s in samples], list(range(8)))
        for sample, row in zip(samples, self.rows):
            self.assertTrue(np.array_equal(sample[3].numpy(), np.asarray(age.features(row, spec), dtype=np.float32)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_configuration_roundtrip_and_fixed_baseline_scope(self):
        baseline = json.loads((ROOT/'configs/R120-SCIENCE.json').read_text())['job']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            def load(value):
                path.write_text(json.dumps({'jobs': [value]}))
                return config.load_config(path, environ={})['jobs'][0]
            omitted, explicit = load(baseline), load({**baseline, 'training_photometric': None})
            self.assertNotIn('training_photometric', config.job_contract(omitted))
            self.assertIsNone(config.job_contract(explicit)['training_photometric'])
            try:
                accepted = load({**baseline, 'training_photometric': POLICY})
            except ValueError as error:
                self.fail(f'fixed gamma configuration not accepted: {error}')
            self.assertEqual(config.job_contract(accepted)['training_photometric'], POLICY)
            for delta in ({'training_photometric': 'gamma-reciprocal05-v1'},
                          {'lr_decay_updates': 8000}, {'backbone_lr': 5e-5},
                          {'loss': None}, {'image_size': 768, 'prepared_size': 768}):
                with self.subTest(delta=delta), self.assertRaises(ValueError):
                    load({**baseline, 'training_photometric': POLICY, **delta})


if __name__ == '__main__':
    unittest.main()
