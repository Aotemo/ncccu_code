"""Three synthetic checks for the optional training photometric policy.

Parent A runs these on the guarded Pod against the frozen runtime Pillow.
The fixtures are pixel oracles for the legacy sample-key augmentation stream,
not a second augmentation platform.
"""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import data

BaseRandom = random.Random
IMAGE_SIZE = 32
SCALE = (.85, 1.0)
SEED = 20261029


class TracedRandom(BaseRandom):
    def __init__(self, seed, forced_photometric=None):
        super().__init__(seed)
        self.random_calls = 0
        self.bounds = []
        self.forced_photometric = list(forced_photometric or [])
        self.photometric_draws = 0

    def random(self):
        self.random_calls += 1
        return super().random()

    def uniform(self, a, b):
        self.bounds.append((a, b))
        value = super().uniform(a, b)
        # Advance the state exactly as the real path would, then return a forced
        # factor so the pixel-difference assertion cannot depend on the seed.
        if (a, b) == (.9, 1.1) and self.photometric_draws < len(self.forced_photometric):
            self.photometric_draws += 1
            return self.forced_photometric[self.photometric_draws - 1]
        return value


def finish(image, image_size, scale, dx, dy):
    from PIL import Image
    import numpy as np
    import torch

    ratio = image_size * scale / max(image.size)
    target = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    image = image.resize(target, Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (image_size, image_size))
    canvas.paste(image, (round((image_size - target[0]) * dx), round((image_size - target[1]) * dy)))
    tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float().div_(255)
    mean, std = (.485, .456, .406), (.229, .224, .225)
    return tensor.sub_(tensor.new_tensor(mean)[:, None, None]).div_(tensor.new_tensor(std)[:, None, None])


def legacy_render(image, image_size, seed, scale_range):
    """Independent copy of the pre-policy training path, including all 8 draws."""
    from PIL import Image, ImageEnhance, ImageOps

    rng = random.Random(seed)
    scale, dx, dy = 1.0, .5, .5
    if rng.random() < .5:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.9, 1.1))
    image = ImageEnhance.Color(image).enhance(rng.uniform(.9, 1.1))
    image = image.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR,
                         expand=True, fillcolor=(0, 0, 0))
    scale, dx, dy = rng.uniform(*scale_range), rng.uniform(.25, .75), rng.uniform(.25, .75)
    return finish(image, image_size, scale, dx, dy)


def geometry_render(image, image_size, seed, scale_range):
    """Same stream and geometry without Brightness/Contrast/Color application."""
    from PIL import Image, ImageOps

    rng = random.Random(seed)
    scale, dx, dy = 1.0, .5, .5
    if rng.random() < .5:
        image = ImageOps.mirror(image)
    for _ in range(3):
        rng.uniform(.9, 1.1)
    image = image.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR,
                         expand=True, fillcolor=(0, 0, 0))
    scale, dx, dy = rng.uniform(*scale_range), rng.uniform(.25, .75), rng.uniform(.25, .75)
    return finish(image, image_size, scale, dx, dy)


class PhotometricPolicyTest(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'train').mkdir()
        # Non-constant RGB: any applied photometric factor must move pixels.
        self.image = Image.new('RGB', (40, 28))
        self.image.putdata([(20 + x * 5 % 220, 15 + y * 7 % 230, 25 + (x * y) % 210)
                            for y in range(28) for x in range(40)])
        self.row = {'name': 'synthetic.png', 'relative_path': 'train/synthetic.png',
                    'sha256': 'a' * 64, 'label': 1}

    def make(self, training=True, **kwargs):
        with patch.object(data, 'configured_cache', return_value=None):
            return data.ImageDataset(self.root, [self.row], IMAGE_SIZE, training=training,
                                     training_scale=list(SCALE), **kwargs)

    def render(self, seed=SEED, forced_photometric=None, **kwargs):
        traces = []

        def factory(value):
            rng = TracedRandom(value, forced_photometric)
            traces.append((value, rng))
            return rng

        dataset = self.make(**kwargs)
        with patch.object(dataset, '_read_image', return_value=self.image.copy()), \
                patch.object(data.random, 'Random', side_effect=factory):
            result = dataset[(0, seed)]
        return result, traces

    def test_default_none_matches_legacy_pixels_and_eight_draws(self):
        import torch
        actual, traces = self.render()
        expected = legacy_render(self.image, IMAGE_SIZE, SEED, SCALE)
        self.assertTrue(torch.equal(actual[0], expected))
        self.assertEqual(len(traces), 1)
        rng = traces[0][1]
        # One direct random() for the mirror plus seven uniform() calls, each of
        # which consumes one primitive draw internally: 8 in total.
        self.assertEqual(rng.random_calls, 8)
        self.assertEqual(rng.bounds, [(.9, 1.1)] * 3 + [(-15, 15), SCALE, (.25, .75), (.25, .75)])
        reference = random.Random(SEED)
        reference.random()
        for low, high in rng.bounds:
            reference.uniform(low, high)
        self.assertEqual(rng.getstate(), reference.getstate())
        self.assertEqual(rng.random(), reference.random())

    def test_identity_consumes_photometric_draws_but_applies_only_geometry(self):
        import torch
        forced = (.5, 1.5, .5)
        legacy, legacy_rng = self.render(forced_photometric=forced)
        actual, traces = self.render(training_photometric='identity-v1', forced_photometric=forced)
        expected = geometry_render(self.image, IMAGE_SIZE, SEED, SCALE)
        self.assertEqual(self.make(training_photometric='identity-v1').training_photometric, 'identity-v1')
        self.assertTrue(torch.equal(actual[0], expected))
        # Identical geometry and stream: the legacy branch applies the forced
        # factors, the identity branch only consumes them.
        self.assertFalse(torch.equal(actual[0], legacy[0]))
        rng = traces[0][1]
        # One direct random() plus seven uniform() calls; Random.uniform consumes
        # one primitive draw itself, so the identity branch still makes 8 draws.
        self.assertEqual(rng.random_calls, 8)
        self.assertEqual(rng.bounds, legacy_rng[0][1].bounds)
        self.assertEqual(rng.getstate(), legacy_rng[0][1].getstate())
        self.assertEqual(rng.random(), legacy_rng[0][1].random())

    def test_eval_ignores_policy_without_random_or_enhancer_and_keeps_schema(self):
        import torch
        from PIL import ImageEnhance
        legacy = self.make(training=False)
        policy = self.make(training=False, training_photometric='identity-v1')
        failure = AssertionError('eval must not create augmentation randomness or enhancers')
        with patch.object(data.random, 'Random', side_effect=failure), \
                patch.object(ImageEnhance, 'Brightness', side_effect=failure), \
                patch.object(ImageEnhance, 'Contrast', side_effect=failure), \
                patch.object(ImageEnhance, 'Color', side_effect=failure):
            with patch.object(legacy, '_read_image', return_value=self.image.copy()):
                default = legacy[0]
            with patch.object(policy, '_read_image', return_value=self.image.copy()):
                alternative = policy[0]
        self.assertTrue(torch.equal(default[0], alternative[0]))
        self.assertEqual(default[1:], (1., 0))
        self.assertEqual(alternative[1:], (1., 0))

        unlabeled = self.make(training=False, include_labels=False)
        with patch.object(unlabeled, '_read_image', return_value=self.image.copy()):
            sample = unlabeled[0]
        self.assertEqual(len(sample), 2)
        self.assertEqual(sample[1], 0)
        self.assertEqual(tuple(sample[0].shape), (3, IMAGE_SIZE, IMAGE_SIZE))


if __name__ == '__main__':
    unittest.main()
