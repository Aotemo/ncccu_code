"""Only a complete selected internal AUROC can admit the one official readout."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex
from retrain import single91


class OfficialGateTest(unittest.TestCase):
    def test_selected_internal_auroc_boundary(self):
        policy = dict(ex.OFFICIAL_GATE_0996)
        self.assertFalse(ex.official_allowed_for_selected(policy, {'auroc': 0.995999999}))
        self.assertTrue(ex.official_allowed_for_selected(policy, {'auroc': 0.996}))
        self.assertTrue(ex.official_allowed_for_selected(policy, {'auroc': 1.0}))
        self.assertTrue(ex.official_allowed_for_selected(None, {'auroc': 0.5}))
        for value in ({}, {'auroc': float('nan')}, {'auroc': True}):
            with self.assertRaises(ValueError):
                ex.official_allowed_for_selected(policy, value)

    def test_below_threshold_freeze_cannot_reserve_official(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'official'
            with patch.object(single91, 'validate_freeze', return_value={'official_evaluations_allowed': 0}):
                with self.assertRaisesRegex(ValueError, 'deferred by frozen policy'):
                    single91.reserve_official(Path(directory) / 'freeze.json', output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
