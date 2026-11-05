"""Synthetic checks for the aggregate-only submission comparison."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/diagnostics/test_submission_aggregate.py'
spec = importlib.util.spec_from_file_location('test_submission_aggregate', SOURCE)
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def write_submission(path, probabilities, labels=None):
    labels = labels or [1 if value >= .5 else 0 for value in probabilities]
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(aggregate.COLUMNS)
        for index, (probability, label) in enumerate(zip(probabilities, labels)):
            writer.writerow(['test_' + str(index).zfill(5) + '.jpg', repr(float(probability)), label])


class SubmissionAggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test-submission-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_alignment_rows_and_threshold_agreement(self):
        left, right = self.root / 'left.csv', self.root / 'right.csv'
        write_submission(left, [.1, .2, .9, .95])
        write_submission(right, [.1, .2, .9, .95])
        first = aggregate.load_submission(left, allow_fixture=True)
        second = aggregate.load_submission(right, allow_fixture=True)
        self.assertEqual(aggregate.require_aligned([first, second]), first['images'])
        wrong = self.root / 'wrong.csv'
        write_submission(wrong, [.1, .2, .9, .95], [0, 0, 1, 0])
        with self.assertRaises(ValueError):
            aggregate.load_submission(wrong, allow_fixture=True)

    def test_misaligned_images_are_rejected(self):
        left, right = self.root / 'left.csv', self.root / 'right.csv'
        write_submission(left, [.1, .2, .9, .95])
        write_submission(right, [.1, .2, .9, .95])
        right_text = right.read_text().replace('test_00001', 'test_99999')
        right.write_text(right_text)
        with self.assertRaises(ValueError):
            aggregate.require_aligned([aggregate.load_submission(left, allow_fixture=True),
                                       aggregate.load_submission(right, allow_fixture=True)])

    def test_average_ranks_and_shift_proportions(self):
        ranks = aggregate.average_ranks(np.asarray([.2, .1, .2, .9]))
        self.assertEqual(ranks.tolist(), [2.5, 1.0, 2.5, 4.0])
        submissions = [{'name': 'a', 'probability': np.asarray([.1, .2, .3, .4]), 'label': np.asarray([0, 0, 0, 1]),
                        'images': ['0', '1', '2', '3']},
                       {'name': 'b', 'probability': np.asarray([.4, .3, .2, .1]), 'label': np.asarray([0, 0, 0, 0]),
                        'images': ['0', '1', '2', '3']}]
        shift = aggregate.rank_shift(submissions)
        self.assertEqual(shift['a|b']['over_10pct'], 1.0)
        correlation = aggregate.pairwise(submissions)
        self.assertAlmostEqual(correlation['a|b']['pearson'], -1.0)
        self.assertAlmostEqual(correlation['a|b']['spearman'], -1.0)

    def test_distribution_and_reference_comparison_are_aggregate_only(self):
        reference = np.linspace(.01, .99, 20)
        probability = np.linspace(.02, .98, 20)
        report = aggregate.build_report(
            [{'name': 'sub', 'probability': probability, 'label': (probability >= .5).astype(int),
              'images': ['x' + str(i) for i in range(20)]}],
            reference=reference, reference_name='sub')
        summary = report['submissions']['sub']
        self.assertEqual(summary['count'], 20)
        self.assertAlmostEqual(summary['threshold_agreement'], 1.0)
        self.assertIn('ecdf_max_difference', report['against_internal_reference'])
        text = repr(report)
        self.assertNotIn('x0', text)
        self.assertNotIn('per_image', text)

    def test_gap_table_reports_plain_differences(self):
        gaps = aggregate.gap_table({'R120': {'source': .98948, 'platform': .9845}})
        self.assertAlmostEqual(gaps['R120']['difference'], -.00498)


if __name__ == '__main__':
    unittest.main()
