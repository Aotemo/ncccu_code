"""R263R1 has the original science but cannot evaluate official data alone."""
import json
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'runtime/bundle'))
from retrain import experiment91


class R263R1IdentityTests(unittest.TestCase):
    def test_new_round_uses_original_science_and_final_ensemble_only_gate(self):
        science = json.loads((root / 'configs/R263-A-SCIENCE.json').read_text())
        site = {'round_id': 'r263-r1-small-mixstyle-seed20261119-pairlog-24k-p4-9010-20261104'}
        self.assertEqual(science['candidate_profile'], experiment91.MIXSTYLE_SMALL_SEED20261119_BINDING)
        self.assertEqual(science['job']['seed'], 20261119)
        self.assertEqual(experiment91.expected_official_policy(site, science),
                         experiment91.OFFICIAL_FINAL_ENSEMBLE_ONLY)
        changed = dict(science, candidate_profile=experiment91.MIXSTYLE_SMALL_SEED20261117_BINDING)
        with self.assertRaises(ValueError):
            experiment91.expected_official_policy(site, changed)


if __name__ == '__main__':
    unittest.main()
