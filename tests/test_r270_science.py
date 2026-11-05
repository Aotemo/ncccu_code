"""R270 preserves the R120 sigmoid recipe while binding new-seed 768 input."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'runtime/bundle'))
from retrain import experiment91 as ex

ROUND = 'r270-small-global768-seed20261122-pairsigmoid-24k-p4-9010-20261105'
PROFILE = 'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261122-scale085-pairsigmoid-global768-v1'


class R270ScienceTests(unittest.TestCase):
    def fixture(self):
        value = json.loads((root / 'configs/R120-SCIENCE.json').read_text())
        historical = json.loads((root / 'configs/R152-SCIENCE.json').read_text())
        value['candidate_profile'] = PROFILE
        value['candidate'].update(
            id='r270-convnext-small-imagenet22k-to-1k-24k-p4-seed20261122-scale085-pairsigmoid-global768-9010',
            experiment_order=180)
        value['job'].update(seed=20261122, image_size=768, prepared_size=768)
        value['paths']['prepared_cache'] = historical['paths']['prepared_cache']
        value['prepared_cache_receipt'] = copy.deepcopy(historical['prepared_cache_receipt'])
        value['official_evaluation'] = copy.deepcopy(ex.OFFICIAL_FINAL_ENSEMBLE_ONLY)
        return value

    def validate_pinned(self, value):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'SCIENCE.json'
            path.write_text(json.dumps(value))
            site = {'work_root': temporary, 'round_id': ROUND,
                    'science': {'configuration': {'path': str(path),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
            with mock.patch.object(ex, 'load_site', return_value=site):
                return ex.science_configuration()

    def test_registered_seed_sigmoid_global_768_and_official_isolation(self):
        value = self.fixture()
        self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
        self.assertEqual(self.validate_pinned(value), value)
        self.assertEqual(ex.expected_official_policy({'round_id': ROUND}, value),
                         ex.OFFICIAL_FINAL_ENSEMBLE_ONLY)

    def test_scientific_drift_is_rejected(self):
        for field, wrong in (('seed', 20261123), ('image_size', 512),
                             ('prepared_size', 512), ('loss', ex.PAIR_LOGISTIC_LOSS),
                             ('positive_per_batch', 9), ('backbone_lr', 0.0002)):
            value = self.fixture()
            value['job'][field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_pinned(value)
                ex.candidate_from_configuration(value)

    def test_missing_cache_receipt_is_rejected(self):
        value = self.fixture()
        del value['prepared_cache_receipt']
        with self.assertRaises(ValueError):
            self.validate_pinned(value)


if __name__ == '__main__':
    unittest.main()
