"""One small Base mapping fixture; parent A executes it under the guarded CPU channel."""
import sys
from pathlib import Path
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools/pretraining'
sys.path.insert(0, str(TOOLS))

from strict_load import target_key
from base_strict import DEPTHS


class BaseMappingTest(unittest.TestCase):
    def test_public_base_keys_map_to_torchvision_with_gamma(self):
        cases = [
            ('stem.0.weight', 'features.0.0.weight', False),
            ('stem.1.bias', 'features.0.1.bias', False),
            ('stages.1.downsample.1.weight', 'features.2.1.weight', False),
            ('stages.2.blocks.26.mlp.fc2.bias', 'features.5.26.block.5.bias', False),
            ('stages.3.blocks.2.gamma', 'features.7.2.layer_scale', True),
            ('head.norm.weight', 'classifier.0.weight', False),
            ('head.fc.bias', 'classifier.2.bias', False),
        ]
        for source, target, gamma in cases:
            self.assertEqual(target_key(source, DEPTHS), (target, gamma), source)

    def test_depth_and_downsample_bounds_rejected(self):
        with self.assertRaises(ValueError):
            target_key('stages.2.blocks.27.gamma', DEPTHS)
        with self.assertRaises(ValueError):
            target_key('stages.0.blocks.3.conv_dw.weight', DEPTHS)
        with self.assertRaises(ValueError):
            target_key('stages.1.downsample.2.weight', DEPTHS)

    def test_base_block_depths(self):
        self.assertEqual(DEPTHS, (3, 3, 27, 3))
        self.assertEqual(sum(DEPTHS), 36)


if __name__ == '__main__':
    unittest.main()
