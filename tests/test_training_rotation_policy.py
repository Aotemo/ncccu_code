"""Two synthetic checks for the optional training rotation policy.

Parent A runs these on the guarded Pod.  The oracles are independent copies of
the legacy augmentation arithmetic, not calls back into the dataset branch.
"""
from contextlib import nullcontext
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
BOUNDS = [(.9, 1.1)] * 3 + [(-15, 15), SCALE, (.25, .75), (.25, .75)]


class TracedRandom(BaseRandom):
    def __init__(self, seed):
        super().__init__(seed)
        self.random_calls = 0
        self.bounds = []

    def random(self):
        self.random_calls += 1
        return super().random()

    def uniform(self, a, b):
        self.bounds.append((a, b))
        return super().uniform(a, b)


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
    from PIL import Image, ImageEnhance, ImageOps

    rng = BaseRandom(seed)
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


def identity_render(image, image_size, seed, scale_range):
    """Mirror and photometric steps apply; the angle is drawn but never rotated."""
    from PIL import Image, ImageEnhance, ImageOps

    rng = BaseRandom(seed)
    scale, dx, dy = 1.0, .5, .5
    if rng.random() < .5:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.9, 1.1))
    image = ImageEnhance.Color(image).enhance(rng.uniform(.9, 1.1))
    rng.uniform(-15, 15)
    scale, dx, dy = rng.uniform(*scale_range), rng.uniform(.25, .75), rng.uniform(.25, .75)
    return finish(image, image_size, scale, dx, dy)


class TrainingRotationPolicyTest(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'train').mkdir()
        self.image = Image.new('RGB', (40, 28))
        self.image.putdata([(18 + x * 7 % 220, 11 + y * 9 % 230, 27 + (x * y) % 200)
                            for y in range(28) for x in range(40)])
        self.row = {'name': 'synthetic.png', 'relative_path': 'train/synthetic.png',
                    'sha256': 'a' * 64, 'label': 1, 'age': '51'}

    def make(self, training=True, rotation=None):
        with patch.object(data, 'configured_cache', return_value=None):
            return data.ImageDataset(self.root, [self.row], IMAGE_SIZE, training=training,
                                     training_scale=list(SCALE), training_rotation=rotation)

    def render(self, dataset, seed=SEED, guard_rotate=False):
        from PIL import Image
        traces = []

        def factory(value):
            rng = TracedRandom(value)
            traces.append((value, rng))
            return rng

        key = (0, seed) if dataset.training else 0
        guard = (patch.object(Image.Image, 'rotate',
                              side_effect=AssertionError('rotation must be skipped'))
                 if guard_rotate else nullcontext())
        with patch.object(dataset, '_read_image', return_value=self.image.copy()), \
                patch.object(data.random, 'Random', side_effect=factory), guard:
            result = dataset[key]
        return result, traces

    def replay(self, rng):
        reference = BaseRandom(SEED)
        reference.random()
        for low, high in rng.bounds:
            reference.uniform(low, high)
        self.assertEqual(rng.getstate(), reference.getstate())
        self.assertEqual(rng.random(), reference.random())

    def test_default_none_matches_legacy_pixels_and_eight_draws(self):
        import torch
        before = self.image.tobytes()
        actual, traces = self.render(self.make())
        expected = legacy_render(self.image, IMAGE_SIZE, SEED, SCALE)
        self.assertTrue(torch.equal(actual[0], expected))
        self.assertEqual(self.image.tobytes(), before)
        self.assertEqual(len(traces), 1)
        rng = traces[0][1]
        self.assertEqual(rng.random_calls, 8)
        self.assertEqual(rng.bounds, BOUNDS)
        self.replay(rng)

    def test_identity_skips_rotate_with_same_draws_and_eval_pixels(self):
        import torch
        actual, traces = self.render(self.make(rotation='identity-v1'), guard_rotate=True)
        expected = identity_render(self.image, IMAGE_SIZE, SEED, SCALE)
        self.assertTrue(torch.equal(actual[0], expected))
        rng = traces[0][1]
        self.assertEqual(rng.random_calls, 8)
        self.assertEqual(rng.bounds, BOUNDS)
        self.replay(rng)
        self.assertEqual(self.make(rotation='identity-v1').training_rotation, 'identity-v1')

        failure = AssertionError('eval must not create augmentation randomness or rotate')
        legacy, policy = self.make(training=False), self.make(training=False, rotation='identity-v1')
        from PIL import Image
        with patch.object(data.random, 'Random', side_effect=failure), \
                patch.object(Image.Image, 'rotate', side_effect=failure):
            with patch.object(legacy, '_read_image', return_value=self.image.copy()):
                default = legacy[0]
            with patch.object(policy, '_read_image', return_value=self.image.copy()):
                alternative = policy[0]
        self.assertTrue(torch.equal(default[0], alternative[0]))
        self.assertEqual(default[1:], (1., 0))
        self.assertEqual(alternative[1:], (1., 0))


if __name__ == '__main__':
    unittest.main()
