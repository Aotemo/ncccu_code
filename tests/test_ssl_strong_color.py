"""Independent constant-pixel oracles and fixed SSL-only science contracts."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
POLICY = 'ssl-strong-color-v1'


class StrongColorTest(unittest.TestCase):
    def test_pixels_rng_and_SSL_only_consumer(self):
        import numpy as np
        import torch
        from PIL import Image
        from retrain import data
        self.assertTrue(hasattr(data, '_ssl_strong_color'), 'fixed SSL strong color helper missing')
        # A gray-only view has a closed-form luminance, independent of jitter order.
        def gray_only(seed):
            rng = random.Random(data._seed(seed, POLICY))
            return rng.random() >= .8 and rng.random() < .2
        gray_seed = next(seed for seed in range(4096) if gray_only(seed))
        image = Image.new('RGB', (9, 7), (100, 150, 200))
        original = image.tobytes()
        self.assertEqual(set(data._ssl_strong_color(image, gray_seed).tobytes()), {141})
        # Contrast/saturation/hue cannot change constant achromatic pixels;
        # only the declared sampled multiplicative brightness can change gray100.
        for seed in range(16):
            rng = random.Random(data._seed(seed, POLICY))
            if rng.random() < .8:
                order = list(range(4)); rng.shuffle(order)
                brightness = rng.uniform(.2, 1.8)
                expected = int(100 * brightness)
            else:
                expected = 100
            result = data._ssl_strong_color(Image.new('RGB', (4, 3), (100, 100, 100)), seed)
            self.assertEqual(set(result.tobytes()), {expected})
        data._ssl_strong_color(image, 3)  # dependency imports precede RNG snapshot
        py_state = random.getstate(); np_state = np.random.get_state(); th_state = torch.get_rng_state()
        outputs = [data._ssl_strong_color(image, seed) for seed in range(16)]
        self.assertEqual(random.getstate(), py_state)
        actual_np = np.random.get_state()
        self.assertEqual(actual_np[0], np_state[0]); np.testing.assert_array_equal(actual_np[1], np_state[1])
        self.assertEqual(actual_np[2:], np_state[2:]); self.assertTrue(torch.equal(torch.get_rng_state(), th_state))
        self.assertEqual(image.tobytes(), original)
        self.assertGreater(len({value.tobytes() for value in outputs}), 1)
        for seed, value in enumerate(outputs):
            self.assertEqual((value.mode, value.size), ('RGB', image.size))
            self.assertEqual(value.tobytes(), data._ssl_strong_color(image, seed).tobytes())
        kwargs = dict(training=True, normalization='imagenet', training_views=2,
                      include_labels=False, training_photometric=POLICY, training_scale=[.85, 1.])
        dataset = data.ImageDataset('/', [], 32, **kwargs)
        tensor = dataset._image_tensor(image, 3)
        self.assertEqual(tuple(tensor.shape), (3, 32, 32)); self.assertTrue(torch.isfinite(tensor).all())
        for changed in ({'training': False}, {'include_labels': True}, {'shared_view_geometry': True}):
            with self.assertRaises(ValueError):
                data.ImageDataset('/', [], 32, **dict(kwargs, **changed))

    def test_recipe_contract_and_original_streams(self):
        from retrain import experiment91 as ex, ssl_stage as ssl
        self.assertTrue(hasattr(ssl, 'STRONG_COLOR_RECIPE'), 'fixed strong-color SSL recipe missing')
        expected = dict(ssl.NTXENT_RECIPE, version='fit-ntxent-strongcolor1600-v1', training_photometric=POLICY)
        self.assertEqual(ssl.STRONG_COLOR_RECIPE, expected)
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        for label, seed, order, binding in (
            ('R236', 20261029, 151, ex.FIT_NTXENT_STRONG_COLOR_SMALL_BINDING),
            ('R237', 20261027, 152, ex.FIT_NTXENT_STRONG_COLOR_SMALL_SEED27_BINDING),
        ):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            job = copy.deepcopy(original['job']); job['seed'] = seed
            self.assertEqual(value['job'], job); self.assertIsNone(value['job'].get('training_photometric'))
            self.assertEqual(value['candidate_profile'], binding); self.assertEqual(value['candidate']['experiment_order'], order)
            self.assertEqual(value['ssl_recipe'], expected); self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            for key in ('public_source', 'paths', 'data_split', 'schedule', 'early_stopping', 'structure'):
                self.assertEqual(value[key], original[key])
            current = ssl.ssl_rng_derivation(expected, seed, 8)
            prior = ssl.ssl_rng_derivation(ssl.NTXENT_RECIPE, seed, 8)
            for key in ('learning_domain', 'sampling_domain', 'projector_seed', 'model_seeds', 'sampler_seed', 'loader_seeds'):
                self.assertEqual(current[key], prior[key])
            manifest = {'fit_indices': list(range(128)), 'dev_indices': [128], 'audit_indices': []}
            self.assertEqual(ssl.validate_recipe(expected, job, manifest, 8), expected)
            with self.assertRaises(ValueError):
                ssl.validate_recipe(dict(expected, training_photometric='identity-v1'), job, manifest, 8)
        for label in ('R115', 'R228', 'R230', 'R231', 'R232', 'R233', 'R234', 'R235'):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            ssl.validate_recipe(value['ssl_recipe'], value['job'], manifest, 8)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(max(1, int(os.environ['NCCCU_CPU_THREADS']) // 2))
    torch.set_num_interop_threads(1)
    unittest.main()
