"""Synthetic tests for P3 opposite-label existence and exact-byte counts."""
import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'tools/diagnostics'))
spec = importlib.util.spec_from_file_location('baseline', root / 'tools/diagnostics/p3_baseline_compare.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def row(index, label, phash, dhash, raw):
    return {'global_index': index, 'label': label,
            'hashes': [f'{phash:016x}', f'{dhash:016x}'], 'raw_sha256': raw}


class BaselineTest(unittest.TestCase):
    def test_counts_queries_once_and_exact_source_bytes_separately(self):
        fit = [row(10, 1, 0, 0, 'same'), row(11, 1, 1, 1, 'other')]
        dev = [row(0, 0, 0, 0, 'same'), row(1, 0, 15, 15, 'different'), row(2, 0, 255, 255, 'none')]
        result = baseline.count_opposite_matches(dev, fit)
        self.assertEqual(result['query_count'], 3)
        self.assertEqual(result['matched_query_count'], 2)
        self.assertEqual(result['same_original_sha_query_count'], 1)
        self.assertAlmostEqual(result['matched_fraction'], 2 / 3)

    def test_rejects_incomplete_hashes_and_wrong_labels(self):
        q, t = row(0, 0, 0, 0, 'a'), row(1, 1, 0, 0, 'b')
        with self.assertRaises(ValueError):
            baseline.count_opposite_matches([dict(q, hashes=None)], [t])
        with self.assertRaises(ValueError):
            baseline.count_opposite_matches([q], [dict(t, label=0)])
        with self.assertRaises(ValueError):
            baseline.count_opposite_matches([q, q], [t])


if __name__ == '__main__':
    unittest.main()
