"""One small Swin mapping/static-index fixture; parent A executes it in the guarded Pod."""
import sys
from pathlib import Path
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools/pretraining'
sys.path.insert(0, str(TOOLS))

from swin_strict import DEPTHS, HEADS, WINDOW, relative_position_index, swin_target_key


class SwinSourceTest(unittest.TestCase):
    def test_public_swin_keys_map_to_torchvision(self):
        cases = [
            ('patch_embed.proj.weight', 'features.0.0.weight'),
            ('patch_embed.norm.bias', 'features.0.2.bias'),
            ('layers.0.blocks.1.norm1.weight', 'features.1.1.norm1.weight'),
            ('layers.2.blocks.17.mlp.fc1.weight', 'features.5.17.mlp.0.weight'),
            ('layers.2.blocks.17.mlp.fc2.bias', 'features.5.17.mlp.3.bias'),
            ('layers.3.blocks.1.attn.qkv.bias', 'features.7.1.attn.qkv.bias'),
            ('layers.3.blocks.1.attn.relative_position_bias_table',
             'features.7.1.attn.relative_position_bias_table'),
            ('layers.1.downsample.reduction.weight', 'features.2.reduction.weight'),
            ('layers.2.downsample.norm.bias', 'features.4.norm.bias'),
            ('norm.weight', 'norm.weight'),
            ('head.fc.bias', 'head.bias'),
        ]
        for source, target in cases:
            self.assertEqual(swin_target_key(source), target, source)

    def test_depth_and_downsample_bounds_rejected(self):
        for key in ('layers.0.blocks.2.gamma', 'layers.2.blocks.18.mlp.fc1.weight',
                    'layers.4.blocks.0.norm1.weight', 'layers.0.downsample.reduction.bias',
                    'patch_embed.proj.gamma'):
            with self.assertRaises(ValueError):
                swin_target_key(key)

    def test_deterministic_relative_position_index(self):
        import torch
        index = relative_position_index(WINDOW)
        self.assertEqual(index.dtype, torch.int64)
        self.assertEqual(tuple(index.shape), (WINDOW * WINDOW * WINDOW * WINDOW,))
        self.assertEqual(int(index.min()), 0)
        self.assertEqual(int(index.max()), (2 * WINDOW - 1) ** 2 - 1)
        self.assertEqual(int(index[0]), 84)
        self.assertEqual(int(index[WINDOW * WINDOW - 1]), 0)
        self.assertEqual(int(index[(WINDOW * WINDOW - 1) * WINDOW * WINDOW]), 168)
        self.assertTrue(torch.equal(index, relative_position_index(WINDOW)))

    def test_fixed_swin_structure(self):
        self.assertEqual(DEPTHS, (2, 2, 18, 2))
        self.assertEqual(HEADS, (4, 8, 16, 32))
        self.assertEqual(WINDOW, 7)
        self.assertEqual(sum(DEPTHS), 24)


if __name__ == '__main__':
    unittest.main()
