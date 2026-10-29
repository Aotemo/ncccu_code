"""Two small synthetic-image checks; parent A runs them on the guarded Pod."""
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import data

BaseRandom = random.Random


class TracedRandom(BaseRandom):
    def __init__(self, seed, lower_bound_geometry=False):
        super().__init__(seed)
        self.bounds = []
        self.draws = 0
        self.lower_bound_geometry = lower_bound_geometry

    def random(self):
        self.draws += 1
        return super().random()

    def uniform(self, a, b):
        value = super().uniform(a, b)
        self.bounds.append((a, b))
        if self.lower_bound_geometry:
            # Consume the real draw first, then expose the geometric endpoint.
            if (a, b) == (-15, 15):
                return 0.
            if (a, b) in ((.94, 1.), (.85, 1.)):
                return a
        return value


class TrainingScaleTest(unittest.TestCase):
    def render(self, scale, *, training=True, lower_bound_geometry=False):
        from PIL import Image
        image = Image.new('RGB', (64, 48))
        image.putdata([(80 + x * 2, 80 + y * 3, 80 + (x + y) % 100)
                       for y in range(48) for x in range(64)])
        traces = []
        def factory(seed):
            rng = TracedRandom(seed, lower_bound_geometry)
            traces.append(rng)
            return rng
        with patch.object(data, 'configured_cache', return_value=None):
            dataset = data.ImageDataset('/synthetic', [{'name': 'synthetic.png', 'label': 1}],
                                        64, training, training_scale=scale)
        dataset._read_image = lambda row: image.copy()
        with patch.object(data.random, 'Random', side_effect=factory):
            tensor, label, index = dataset[(0, 20261029)]
        self.assertEqual((label, index), (1., 0))
        return tensor, traces

    def test_default_and_explicit_legacy_range_preserve_pixels_and_random_stream(self):
        import torch
        default, a = self.render(None)
        explicit, b = self.render([.94, 1.])
        self.assertTrue(torch.equal(default, explicit))
        expected = [(.9, 1.1)] * 3 + [(-15, 15), (.94, 1.), (.25, .75), (.25, .75)]
        self.assertEqual(a[0].bounds, expected)
        self.assertEqual(b[0].bounds, expected)
        self.assertEqual(a[0].draws, 8)
        self.assertEqual(b[0].draws, 8)
        self.assertEqual(a[0].getstate(), b[0].getstate())
        self.assertEqual([a[0].random() for _ in range(4)], [b[0].random() for _ in range(4)])

    def test_new_lower_scale_changes_only_geometry_and_eval_remains_identical(self):
        import torch
        old, a = self.render([.94, 1.], lower_bound_geometry=True)
        new, b = self.render([.85, 1.], lower_bound_geometry=True)
        self.assertEqual(a[0].bounds[:4] + a[0].bounds[5:], b[0].bounds[:4] + b[0].bounds[5:])
        self.assertEqual((a[0].bounds[4], b[0].bounds[4]), ((.94, 1.), (.85, 1.)))
        self.assertEqual((a[0].draws, b[0].draws), (8, 8))
        self.assertEqual(a[0].getstate(), b[0].getstate())
        self.assertEqual([a[0].random() for _ in range(4)], [b[0].random() for _ in range(4)])
        self.assertFalse(torch.equal(old, new))
        def content_size(tensor):
            mean = tensor.new_tensor((.485, .456, .406))[:, None, None]
            std = tensor.new_tensor((.229, .224, .225))[:, None, None]
            mask = (tensor * std + mean).amax(dim=0) > 1e-6
            ys, xs = mask.nonzero(as_tuple=True)
            return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        self.assertEqual(content_size(old), (60, 45))
        self.assertEqual(content_size(new), (54, 41))
        eval_default, c = self.render(None, training=False)
        eval_legacy, d = self.render([.94, 1.], training=False)
        eval_new, e = self.render([.85, 1.], training=False)
        self.assertTrue(torch.equal(eval_default, eval_legacy) and torch.equal(eval_default, eval_new))
        self.assertEqual((c, d, e), ([], [], []))


if __name__ == '__main__':
    unittest.main()
