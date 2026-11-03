"""R224/R225 keep the supervised comparison fixed while changing the fit-only signal."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex


class MaskedCandidateTests(unittest.TestCase):
    def test_hog_pair_changes_only_fit_target_and_keeps_R107_supervision(self):
        self.assertTrue(hasattr(ex, 'FIT_HOG_SMALL_BINDING'), 'HOG candidates are not registered')
        from retrain.masked_reconstruction import HOG_RECIPE
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        for label, order, seed, binding in (
                ('R226', 141, 20261029, ex.FIT_HOG_SMALL_BINDING),
                ('R227', 142, 20261027, ex.FIT_HOG_SMALL_SEED27_BINDING)):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            self.assertEqual(value['candidate_profile'], binding)
            self.assertIn(binding, ex.SUPPORTED_BINDINGS)
            expected = copy.deepcopy(original['job']); expected['seed'] = seed
            self.assertEqual(value['job'], expected)
            for key in ('public_source', 'public_asset_receipt', 'paths', 'structure',
                        'data_split', 'schedule', 'early_stopping'):
                self.assertEqual(value[key], original[key])
            self.assertEqual(value['ssl_recipe'], HOG_RECIPE)
            self.assertEqual(value['candidate']['experiment_order'], order)
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            for key, change in [('backbone_lr', 1e-5), ('loss', None), ('image_size', 768), ('training_views', 2)]:
                wrong = copy.deepcopy(value); wrong['job'][key] = change
                with self.subTest(label=label, key=key), self.assertRaises(ValueError):
                    ex.candidate_from_configuration(wrong)

    def test_registered_pair_is_exact_R107_supervision_and_same_public_source(self):
        self.assertTrue(hasattr(ex, 'FIT_MIM_SMALL_BINDING'), 'new masked candidate is not registered')
        from retrain.masked_reconstruction import MASKED_RECIPE
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        for label, order, seed, binding in (
                ('R224', 139, 20261029, ex.FIT_MIM_SMALL_BINDING),
                ('R225', 140, 20261027, ex.FIT_MIM_SMALL_SEED27_BINDING)):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            self.assertEqual(value['candidate_profile'], binding)
            self.assertIn(binding, ex.SUPPORTED_BINDINGS)
            expected = copy.deepcopy(original['job'])
            expected['seed'] = seed
            self.assertEqual(value['job'], expected)
            for key in ('public_source', 'public_asset_receipt', 'paths', 'structure',
                        'data_split', 'schedule', 'early_stopping'):
                self.assertEqual(value[key], original[key])
            self.assertEqual(value['ssl_recipe'], MASKED_RECIPE)
            self.assertEqual(value['candidate']['experiment_order'], order)
            self.assertEqual(value['official_evaluation'], ex.OFFICIAL_GATE_0996)
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            for key, change in [('backbone_lr', 1e-5), ('loss', None), ('image_size', 768), ('training_views', 2)]:
                wrong = copy.deepcopy(value)
                wrong['job'][key] = change
                with self.subTest(label=label, key=key), self.assertRaises(ValueError):
                    ex.candidate_from_configuration(wrong)
            wrong = copy.deepcopy(value)
            wrong['public_source']['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                ex.candidate_from_configuration(wrong)


if __name__ == '__main__':
    unittest.main()
