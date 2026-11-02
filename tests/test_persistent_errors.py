"""Exact ties and dev-only identity must survive error-attribution analysis."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'persistent_errors', Path(__file__).resolve().parents[1] / 'tools/diagnostics/persistent_errors.py')
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class PersistentErrorsTests(unittest.TestCase):
    def test_tied_pairs_conserve_both_class_totals(self):
        # Two positives (0, 2), three negatives (0, 1, 3):
        # P0 contributes 2.5 errors, P2 contributes 1; total = 3.5.
        result = diagnostic.pair_contributions([1, 1, 0, 0, 0], [0., 2., 0., 1., 3.])
        self.assertEqual(result, [5, 2, 1, 2, 4])
        self.assertEqual(sum(result[:2]), 7)
        self.assertEqual(sum(result[2:]), 7)
        # A tied perfect-looking score must still count a half error.
        self.assertEqual(diagnostic.pair_contributions([0, 1], [1., 1.]), [1, 1])

    def test_nonfinite_or_single_class_predictions_are_rejected(self):
        for labels, scores in (([0, 1], [0., float('nan')]),
                               ([1, 1], [0., 1.]), ([0, 2], [0., 1.]),
                               ([0, 1], [0.])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                diagnostic.pair_contributions(labels, scores)

    def test_wrong_split_or_reordered_rows_cannot_become_dev_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = [{'name': 'a.jpg', 'label': 0}, {'name': 'b.jpg', 'label': 1}]
            official = root / 'official.csv'
            official.write_text('name,label,logit\na.jpg,0,0\nb.jpg,1,1\n')
            with self.assertRaises(ValueError):
                diagnostic.read_predictions(official, expected, 'ema')
            path = root / 'predictions/ema/dev-cycle-0004.csv'
            path.parent.mkdir(parents=True)
            path.write_text('name,label,logit\na.jpg,0,0\nb.jpg,1,1\n')
            self.assertEqual(diagnostic.read_predictions(path, expected, 'ema'), [0., 1.])
            path.write_text('name,label,logit\nb.jpg,1,1\na.jpg,0,0\n')
            with self.assertRaises(ValueError):
                diagnostic.read_predictions(path, expected, 'ema')
            path.write_text('name,label,logit\na.jpg,1,0\nb.jpg,1,1\n')
            with self.assertRaises(ValueError):
                diagnostic.read_predictions(path, expected, 'ema')


if __name__ == '__main__':
    unittest.main()
