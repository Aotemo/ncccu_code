"""Independent fixed geometry and blind-seal tests; no task images."""
import unittest
from tools.diagnostics.fit_disc_consensus import choose_fit, consensus_box, box_coverage, validate_annotations


class DiscConsensusTest(unittest.TestCase):
    def test_fit_selection_is_deterministic_disjoint_and_rejects_duplicates(self):
        first = choose_fit(list(range(100)), list(range(10)), 64, 20261112)
        self.assertEqual(first, choose_fit(list(reversed(range(100))), list(range(10)), 64, 20261112))
        self.assertEqual(len(set(first)), 64)
        self.assertFalse(set(first) & set(range(10)))
        with self.assertRaises(ValueError):
            choose_fit([1, 1, 2], [], 2, 1)

    def test_consensus_agreement_and_edge_clamping(self):
        box = consensus_box((512, 512), {'status': 'located', 'center_xy': [70, 250]}, (0, 122, 256, 378))
        self.assertEqual(box['status'], 'located')
        self.assertEqual(box['box'], [0, 122, 256, 378])
        self.assertEqual(consensus_box((512, 512), {'status': 'located', 'center_xy': [450, 250]}, (0, 122, 256, 378))['status'], 'unknown')
        self.assertEqual(consensus_box((512, 512), {'status': 'unknown', 'center_xy': None}, (0, 0, 256, 256))['status'], 'unknown')

    def test_margin_and_unknown_are_separate_failures(self):
        self.assertEqual(box_coverage([0, 0, 256, 256], [80, 80, 128, 128]), (True, True))
        self.assertEqual(box_coverage([0, 0, 256, 256], [5, 80, 53, 128]), (True, False))
        self.assertEqual(box_coverage(None, [80, 80, 128, 128]), (False, False))
        self.assertEqual(box_coverage([0, 0, 256, 256], None), (False, False))

    def test_blind_annotations_require_complete_unique_aliases(self):
        valid = [{'alias': 'B001', 'box': [1, 2, 3, 4]}, {'alias': 'B002', 'box': None}]
        validate_annotations(valid, ['B001', 'B002'])
        for bad in (valid[:1], valid + valid[:1], [{'alias': 'B001', 'box': [4, 2, 3, 4]}, valid[1]]):
            with self.assertRaises(ValueError):
                validate_annotations(bad, ['B001', 'B002'])

if __name__ == '__main__':
    unittest.main()
