"""Focused checks for the preregistered P3 ranking and hash definitions."""
import importlib.util
import json
from pathlib import Path
import tempfile
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

    def test_shared_pair_summary_counts_pairs_unique_negatives_and_support(self):
        value = p3.shared_pair_summary([{1, 3}, {1, 3}, {3}])
        self.assertEqual(value, {'pair_count': 5, 'unique_negative_count': 2,
                                 'negative_count_support_at_least_three': 1})

    def test_private_shared_negative_support_counts_each_positive_once(self):
        self.assertEqual(p3.shared_negative_support([{1, 3}, {1, 3}, {3}]), {1: 2, 3: 3})

    def test_hash_index_matches_bruteforce_including_four_bit_boundary(self):
        values = [(i, ((i * 0x123456789ABCDEF) & ((1 << 64) - 1),
                       (i * 0xFEDCBA987654321) & ((1 << 64) - 1))) for i in range(100)]
        values += [(100, (0b1111, 0b1111)), (101, (0b11111, 0))]
        index = p3.HashIndex(values)
        for _, query in values:
            expected = {i for i, value in values if p3.near_duplicate(query, value)}
            self.assertEqual(set(index.find(query)), expected)
        self.assertIn(100, index.find((0, 0)))
        self.assertNotIn(101, index.find((0, 0)))

    def test_quality_extreme_needs_two_predeclared_flags_and_full_reference(self):
        ref = [{'brightness': float(i), 'contrast': float(i), 'sharpness': float(i), 'field': float(i)}
               for i in range(10, 20)]
        thresholds = p3.quality_thresholds(ref, required=10)
        self.assertTrue(p3.quality_extreme({'brightness': 0., 'contrast': 0., 'sharpness': 15., 'field': 15.}, thresholds))
        self.assertFalse(p3.quality_extreme({'brightness': 0., 'contrast': 15., 'sharpness': 15., 'field': 15.}, thresholds))
        self.assertIsNone(p3.quality_thresholds(ref[:-1], required=10))
        self.assertIsNone(p3.quality_extreme(None, thresholds))

    def test_scan_order_prioritizes_fixed_dev_and_fit_positives_without_duplication(self):
        result = p3.image_scan_order([3, 1, 2], [5, 4, 6], {1: 1, 2: 0, 3: 1}, [1])
        self.assertEqual(result, [('dev', 1, 4), ('fit', None, 1), ('fit', None, 3),
                                  ('dev', 0, 5), ('dev', 2, 6), ('fit', None, 2)])
        with self.assertRaises(ValueError):
            p3.image_scan_order([1, 1], [2], {1: 0}, [])

    def test_scan_resume_rejects_changed_order_and_missing_part(self):
        order = [('dev', 1, 4), ('fit', None, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'parts').mkdir()
            first = root / 'parts/part-000000-000000.json'
            first.write_text(json.dumps({'start': 0, 'rows': [
                {'split': 'dev', 'dev_index': 1, 'global_index': 4}]}))
            self.assertEqual(len(p3.image_parts(root, order)), 1)
            second = root / 'parts/part-000001-000001.json'
            second.write_text(json.dumps({'start': 1, 'rows': [
                {'split': 'fit', 'dev_index': None, 'global_index': 3}]}))
            with self.assertRaises(ValueError):
                p3.image_parts(root, order)


if __name__ == '__main__':
    unittest.main()
