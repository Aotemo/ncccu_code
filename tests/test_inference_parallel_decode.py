"""Parallel batch decode must be bit-identical to serial and keep logits equal."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime' / 'bundle'))

import numpy as np
import torch
from PIL import Image

from retrain import inference
from retrain.data import ImageDataset


def _write_images(folder, count=4):
    folder.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(20261105)
    for index in range(count):
        height, width = 240 + 8 * index, 320 - 4 * index
        pixels = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels, 'RGB').save(folder / f'image_{index:02d}.jpg', quality=90)


class ParallelDecodeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _write_images(self.root / 'test')
        self.rows = [{'name': f'image_{index:02d}.jpg',
                      'relative_path': f'test/image_{index:02d}.jpg',
                      'label': 0, 'age': ''} for index in range(4)]
        self.dataset = ImageDataset(self.root, self.rows, 224, False, 'imagenet', prepared_size=0)

    def test_serial_and_parallel_items_are_bit_identical(self):
        serial = [self.dataset[index] for index in range(len(self.rows))]
        parallel = inference.preprocess_batch(self.dataset, range(len(self.rows)), workers=4)
        self.assertEqual(len(serial), len(parallel))
        for left, right in zip(serial, parallel):
            self.assertTrue(torch.equal(left[0], right[0]))
            self.assertEqual(left[1:], right[1:])

    def test_fixed_model_logits_are_identical(self):
        class MeanModel(torch.nn.Module):
            def forward(self, images, age_features=None):
                return images.mean(dim=(1, 2, 3)).unsqueeze(1)

        model = MeanModel().eval()
        outputs = {}
        for workers in (1, 4):
            items = inference.preprocess_batch(self.dataset, range(len(self.rows)), workers=workers)
            images = torch.stack([item[0] for item in items])
            with torch.inference_mode():
                outputs[workers] = model(images).reshape(-1).clone()
        self.assertTrue(torch.equal(outputs[1], outputs[4]))

    def test_workers_one_matches_plain_loop(self):
        plain = [self.dataset[index] for index in range(len(self.rows))]
        single = inference.preprocess_batch(self.dataset, range(len(self.rows)), workers=1)
        for left, right in zip(plain, single):
            self.assertTrue(torch.equal(left[0], right[0]))


if __name__ == '__main__':
    unittest.main()
