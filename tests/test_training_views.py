"""Three synthetic CPU checks; parent A runs them on the guarded Pod."""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import data
from retrain.classification_loss import classification_loss

BaseRandom = random.Random


class TracedRandom(BaseRandom):
    def __init__(self, seed):
        super().__init__(seed)
        self.draws = 0

    def random(self):
        self.draws += 1
        return super().random()


class TrainingViewsTest(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'train').mkdir()
        self.image = Image.new('RGB', (64, 48))
        self.image.putdata([(70 + x * 2, 60 + y * 3, 80 + (x + y) % 100)
                            for y in range(48) for x in range(64)])
        self.image.save(self.root / 'train/synthetic.png')
        self.row = {'name': 'synthetic.png', 'relative_path': 'train/synthetic.png',
                    'sha256': 'a' * 64, 'label': 1, 'age': 51}

    def render(self, views=None, *, seed=20261029, training=True, cached=False, with_age=False):
        from retrain import age
        traces = []

        def factory(value):
            rng = TracedRandom(value)
            traces.append((value, rng))
            return rng

        kwargs = {} if views is None else {'training_views': views}
        age_spec = {'enabled': True} if with_age else None
        # Age arithmetic has its own contract. Here its boundary is observed so
        # two image views cannot silently duplicate the main sample's features.
        with patch.object(data, 'configured_cache', return_value=None), \
                patch.object(age, 'validate_spec'), \
                patch.object(age, 'features', return_value=[.25, 0.]) as features:
            dataset = data.ImageDataset(self.root, [self.row], 64, training,
                                        age=age_spec, training_scale=[.85, 1.], **kwargs)
            if cached:
                dataset.prepared_cache = Mock()
                dataset.prepared_cache.read.return_value = self.image.copy()
            with patch.object(dataset, '_read_image', wraps=dataset._read_image) as raw_read, \
                    patch.object(data.random, 'Random', side_effect=factory):
                result = dataset[(0, seed)]
            self.assertEqual(raw_read.call_count, 0 if cached else 1)
            if cached:
                dataset.prepared_cache.read.assert_called_once_with(
                    self.row['relative_path'], self.row['sha256'])
            self.assertEqual(result[1:3], (1., 0))
            self.assertEqual(len(result), 4 if with_age else 3)
            if with_age:
                features.assert_called_once_with(self.row, age_spec)
                self.assertEqual(result[3].tolist(), [.25, 0.])
            else:
                features.assert_not_called()
        return result, traces

    def test_training_views_preserve_first_pixels_and_use_one_read_and_derived_seed(self):
        import torch
        seed = 20261029
        second_seed = data._seed(seed, 'training-view', 1)
        for cached, with_age in ((False, False), (True, True)):
            with self.subTest(cached=cached, with_age=with_age):
                options = {'cached': cached, 'with_age': with_age}
                default, a = self.render(**options)
                explicit, b = self.render(1, **options)
                dual, c = self.render(2, **options)
                second, d = self.render(seed=second_seed, **options)
                self.assertEqual(tuple(default[0].shape), (3, 64, 64))
                self.assertEqual(tuple(dual[0].shape), (2, 3, 64, 64))
                self.assertTrue(torch.equal(default[0], explicit[0]))
                self.assertTrue(torch.equal(default[0], dual[0][0]))
                self.assertTrue(torch.equal(second[0], dual[0][1]))
                self.assertEqual([value for value, _ in c], [seed, second_seed])
                self.assertEqual([value for value, _ in a], [seed])
                self.assertEqual([rng.draws for _, rng in a + b + c + d], [8] * 5)
                self.assertEqual(a[0][1].getstate(), b[0][1].getstate())
                self.assertEqual(a[0][1].getstate(), c[0][1].getstate())
                self.assertEqual(d[0][1].getstate(), c[1][1].getstate())

    def test_eval_remains_single_view_with_identical_pixels_and_no_random_generator(self):
        import torch
        default, a = self.render(training=False)
        explicit, b = self.render(1, training=False)
        dual_setting, c = self.render(2, training=False)
        self.assertEqual(tuple(dual_setting[0].shape), (3, 64, 64))
        self.assertTrue(torch.equal(default[0], explicit[0]))
        self.assertTrue(torch.equal(default[0], dual_setting[0]))
        self.assertEqual((a, b, c), ([], [], []))

    def test_flatten_aligns_sample_major_labels_age_and_mean_BCE_gradients(self):
        import torch
        images = torch.tensor([-1.2, .4, 1.1, -.7]).reshape(2, 2, 1, 1, 1).requires_grad_()
        labels = torch.tensor([0., 1.])
        age_features = torch.tensor([[.25, 0.], [-.75, 1.]])
        for views in (None, 1):
            kept = data.flatten_training_views(images, labels, age_features, views)
            for actual, original in zip(kept, (images, labels, age_features)):
                self.assertIs(actual, original)
        flat, targets, ages = data.flatten_training_views(images, labels, age_features, 2)
        self.assertEqual(tuple(flat.shape), (4, 1, 1, 1))
        self.assertTrue(torch.equal(flat.flatten(), images.flatten()))
        self.assertEqual(targets.tolist(), [0., 0., 1., 1.])
        self.assertEqual(ages.tolist(), [[.25, 0.], [.25, 0.], [-.75, 1.], [-.75, 1.]])
        no_age = data.flatten_training_views(images, labels, None, 2)
        self.assertIsNone(no_age[2])
        self.assertTrue(torch.equal(no_age[1], targets))
        actual = classification_loss(flat.flatten(), targets, None)
        reference_images = images.detach().clone().requires_grad_()
        expected = .5 * (classification_loss(reference_images[:, 0].flatten(), labels, None)
                         + classification_loss(reference_images[:, 1].flatten(), labels, None))
        actual_gradient, = torch.autograd.grad(actual, images)
        expected_gradient, = torch.autograd.grad(expected, reference_images)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
