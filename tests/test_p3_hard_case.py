"""Focused checks for the preregistered P3 ranking and hash definitions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
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

    def test_adaptive_workers_obey_declared_latency_and_d_state_policy(self):
        self.assertEqual(p3.p3_next_workers(16, 119, 0.1, 0.0, 0), (16, 1, False))
        self.assertEqual(p3.p3_next_workers(16, 119, 0.1, 0.0, 1), (24, 0, False))
        self.assertEqual(p3.p3_next_workers(16, 119, 2.1, 0.0, 0), (8, 0, False))
        self.assertEqual(p3.p3_next_workers(16, 119, 0.1, 0.25, 0), (8, 0, False))
        self.assertEqual(p3.p3_next_workers(1, 119, 5.1, 0.0, 0), (1, 0, True))
        self.assertEqual(p3.p3_next_workers(16, 119, 0.1, 0.5, 0), (8, 0, True))
        self.assertEqual(p3.p3_next_workers(115, 119, 0.1, 0.0, 1), (119, 0, False))

    def test_adaptive_batch_preserves_order_and_stops_submitting_on_emergency(self):
        def worker(index):
            time.sleep((3 - index % 3) * 0.002)
            return index, 0.01
        results, _, seen = p3.p3_parallel_batch(list(range(12)), worker, 3, lambda _: 0.0)
        self.assertEqual(results, list(range(12)))
        self.assertEqual(seen, 0.0)
        results, _, seen = p3.p3_parallel_batch(list(range(12)), worker, 3, lambda _: 0.5)
        self.assertEqual(results, list(range(3)))
        self.assertEqual(seen, 0.5)

    def test_adaptive_batch_propagates_failed_worker(self):
        def bad(index):
            raise ValueError('fixture failure')
        with self.assertRaisesRegex(ValueError, 'fixture failure'):
            p3.p3_parallel_batch([0], bad, 1, lambda _: 0.0)

    def test_adaptive_rows_match_serial_hash_and_quality(self):
        image = Image.new('RGB', (48, 32), (40, 90, 150))
        class Cache:
            def read(self, relative, expected):
                return image.copy()
        def features(view):
            return {'content_luminance_mean': 42., 'content_luminance_std': 7.,
                    'green_laplacian_variance': 3., 'black_fraction': .2}
        items = [('dev', 2, 0), ('fit', None, 1), ('dev', 4, 2)]
        rows = [{'relative_path': 'train/example', 'sha256': 'a' * 64, 'label': i % 2} for i in range(3)]
        def scan(item):
            return p3.image_scan_row(item, rows[item[2]], Cache(), Path('/fixture'),
                                     lambda source, size: source, lambda *args: None, features)
        serial = [scan(item)[0] for item in items]
        parallel, _, _ = p3.p3_parallel_batch(items, scan, 2, lambda _: 0.0)
        self.assertEqual(parallel, serial)
        self.assertEqual(serial[0]['quality']['field'], .8)
        self.assertIsNone(serial[1]['quality'])


if __name__ == '__main__':
    unittest.main()
