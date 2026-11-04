"""Focused checks for the preregistered P3 ranking and hash definitions."""
import importlib.util
from pathlib import Path
import unittest

from PIL import Image

here = Path(__file__).resolve()
source = here.parent / 'p3_hard_case.py'
if not source.exists():
    source = here.parents[1] / 'tools/diagnostics/p3_hard_case.py'
spec = importlib.util.spec_from_file_location('p3_hard_case', source)
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)


class HardCaseTests(unittest.TestCase):
    def test_ties_have_half_weight_in_rank_and_pair_mass(self):
        self.assertEqual(p3.midrank_quantile(2.0, [1.0, 2.0, 2.0, 3.0]), 0.5)
        self.assertEqual(p3.negative_above_mass(2.0, [1.0, 2.0, 2.0, 3.0]), 2.0)

    def test_seven_member_consensus_keeps_partial_disagreement_distinct(self):
        self.assertEqual(p3.consensus(7, 7), 'all')
        self.assertEqual(p3.consensus(3, 7), 'mixed')
        self.assertEqual(p3.consensus(0, 7), 'none')
        with self.assertRaises(ValueError):
            p3.consensus(8, 7)

    def test_identical_image_hashes_and_double_threshold(self):
        image = Image.new('RGB', (64, 64), (80, 90, 100))
        hashes = p3.image_hashes(image)
        self.assertEqual(hashes, p3.image_hashes(image.copy()))
        self.assertTrue(p3.near_duplicate(hashes, hashes))
        self.assertFalse(p3.near_duplicate(hashes, (hashes[0] ^ 31, hashes[1])))
        self.assertFalse(p3.near_duplicate(hashes, (hashes[0], hashes[1] ^ 31)))

    def test_common_negative_requires_strict_misordering_by_every_model(self):
        count = p3.shared_negative_count([0.5, 0.5], [[0.6, 0.7, 0.5], [0.2, 0.8, 0.9]])
        self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
