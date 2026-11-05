"""R271 changes only the registered R201-family seed and member identity."""
import copy
import json
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'runtime/bundle'))
from retrain import experiment91 as ex


class R271ScienceTests(unittest.TestCase):
    def fixture(self):
        value = json.loads((root / 'configs/R259-B-SCIENCE.json').read_text())
        value['candidate_profile'] = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261123-scale085-pairsigmoid-wide2-peakgrad-v1'
        value['candidate']['id'] = 'r271-convnext-small-imagenet22k-to-1k-24k-p4-seed20261123-scale085-pairsigmoid-wide2-peakgrad-9010'
        value['candidate']['experiment_order'] = 181
        value['job']['seed'] = 20261123
        return value

    def test_new_seed_identity_retains_registered_wide2_recipe(self):
        value = self.fixture()
        try:
            candidate = ex.candidate_from_configuration(value)
        except ValueError as error:
            self.fail('registered new identity was rejected: ' + str(error))
        self.assertEqual(candidate, value['candidate'])
        site = {'round_id': 'r271-small-wide2-seed20261123-pairsigmoid-24k-p4-9010-20261105'}
        self.assertEqual(ex.expected_official_policy(site, value), ex.OFFICIAL_FINAL_ENSEMBLE_ONLY)

    def test_wrong_seed_or_loss_is_rejected(self):
        for field, bad in (('seed', 20261122), ('loss', None)):
            value = copy.deepcopy(self.fixture()); value['job'][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                ex.candidate_from_configuration(value)


if __name__ == '__main__':
    unittest.main()
