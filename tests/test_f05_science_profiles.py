"""The three OOF rounds reuse R120 science with distinct fold identities."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex


class ThreeFoldScienceTests(unittest.TestCase):
    def test_registered_profiles_keep_r120_job_and_forbid_member_official(self):
        base = json.loads((ROOT / 'configs/R120-SCIENCE.json').read_text())
        for number in range(3):
            label, order = 'R' + str(265 + number), 175 + number
            profile = ('single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-'
                       f'seed20261029-scale085-pairsigmoid-f05fold{number}-v1')
            round_id = (f'{label.lower()}-small-f05fold{number}-seed20261029-'
                        'pairsigmoid-24k-p4-9010-20261105')
            with self.subTest(fold=number):
                self.assertEqual(ex.PAIR_SIGMOID_SEED_REPLICAS.get(profile), (label, 20261029, order))
                self.assertIn(profile, ex.SUPPORTED_BINDINGS)
                value = copy.deepcopy(base)
                value['candidate_profile'] = profile
                value['candidate'].update(
                    id=(f'{label.lower()}-convnext-small-imagenet22k-to-1k-24k-p4-'
                        'seed20261029-scale085-pairsigmoid-9010'), experiment_order=order)
                self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
                self.assertEqual(value['job'], base['job'])
                self.assertEqual(ex.expected_official_policy({'round_id':round_id}, value),
                                 ex.OFFICIAL_FINAL_ENSEMBLE_ONLY)
                with self.assertRaises(ValueError):
                    ex.expected_official_policy({'round_id':round_id}, base)
                for pod in ('A', 'B'):
                    configured = json.loads((ROOT / 'configs' / f'{label}-{pod}-SCIENCE.json').read_text())
                    self.assertEqual(configured['candidate_profile'], profile)
                    self.assertEqual(configured['candidate'], value['candidate'])
                    self.assertEqual(configured['job'], base['job'])
                    self.assertEqual(configured['initial_model_tensor_sha256'], base['initial_model_tensor_sha256'])
                    self.assertEqual(ex.validate_fold_configuration({'round_id':round_id}, configured), number)
                    self.assertEqual(ex.expected_official_policy({'round_id':round_id}, configured),
                                     configured['official_evaluation'])


if __name__ == '__main__':
    unittest.main()
