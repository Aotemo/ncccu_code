"""The two explicit R50 identities share all parameters and learning settings."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex


class R50BNBindingTest(unittest.TestCase):
    def test_matched_pair_changes_only_BN_initialization_and_identity(self):
        left = json.loads((ROOT / 'configs/R111-SCIENCE.json').read_text())
        right = json.loads((ROOT / 'configs/R112-SCIENCE.json').read_text())
        for value, order, kind in ((left, 21, 'public'), (right, 22, 'fit_fixed_reference')):
            actual = ex.candidate_from_configuration(value)
            self.assertEqual((actual['experiment_order'], actual['bn_initialization']), (order, kind))
            self.assertEqual((actual['features'], actual['batch_norm_modules'], actual['batch_norm_buffers']), (2048, 53, 159))
            self.assertEqual(value['job']['seed'], 20261029)
            self.assertEqual(value['job']['head_seed'], 20261027)
            self.assertIsNone(value['job'].get('loss'))
            self.assertNotIn('training_scale', value['job'])
            self.assertNotIn('training_views', value['job'])
        for value in (left, right):
            value.pop('candidate_profile')
            for name in ('initial_model_tensor_sha256', 'initial_buffer_sha256'):
                value.pop(name)
            for name in ('id', 'experiment_order', 'bn_initialization'):
                value['candidate'].pop(name)
            value['job'].pop('bn_statistics')
        self.assertEqual(left, right)

    def test_normalization_pins_real_immutable_BN_digest_for_either_arm(self):
        for number in (111, 112):
            science = json.loads((ROOT / ('configs/R'+str(number)+'-SCIENCE.json')).read_text())
            science['initial_buffer_sha256'] = 'a' * 64
            value = {'candidate': science['candidate'], 'batch_norm_modules': 53, 'buffer_count': 159,
                'buffer_sha256': 'a' * 64, 'layer_norms': [], 'stochastic_depth': [],
                'head_in_features': 2048, 'head_out_features': 1, 'dropout': .2,
                'batch_norms': [{'name': 'backbone.bn'+str(i), 'eps': 1e-5,
                                'training': False, 'track_running_stats': True} for i in range(53)]}
            with patch.object(ex, 'candidate', return_value=science['candidate']), \
                    patch.object(ex, 'science_configuration', return_value=science), \
                    patch.object(ex, 'structure_contract', return_value=science['structure']):
                self.assertTrue(ex.validate_normalization(value))
                changed = copy.deepcopy(value);changed['buffer_sha256'] = 'b' * 64
                with self.assertRaises(ValueError):ex.validate_normalization(changed)


if __name__ == '__main__':unittest.main()
