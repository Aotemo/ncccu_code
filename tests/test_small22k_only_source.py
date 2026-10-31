"""One synthetic mapping/config-head check; A runs it on the guarded Pod."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'pretraining'))
import small22k_only_strict as strict
import small_source
import small_strict


def source_keys():
    keys = ['stem.0.weight', 'stem.0.bias', 'stem.1.weight', 'stem.1.bias']
    for stage in (1, 2, 3):
        keys += [f'stages.{stage}.downsample.0.weight', f'stages.{stage}.downsample.0.bias',
                 f'stages.{stage}.downsample.1.weight', f'stages.{stage}.downsample.1.bias']
    for stage, depth in enumerate(strict.DEPTHS):
        for block in range(depth):
            prefix = f'stages.{stage}.blocks.{block}'
            for name in ('conv_dw', 'norm', 'mlp.fc1', 'mlp.fc2'):
                keys += [prefix + '.' + name + '.weight', prefix + '.' + name + '.bias']
            keys.append(prefix + '.gamma')
    keys += ['head.norm.weight', 'head.norm.bias', 'head.fc.weight', 'head.fc.bias']
    return keys


class Small22KOnlySourceTest(unittest.TestCase):
    def test_full_k_head_mapping_and_config_identity(self):
        num_classes = 7
        source, expected, converted = {}, {}, {}
        for key in source_keys():
            gamma = key.endswith('.gamma')
            if key == 'head.fc.weight':
                value = torch.zeros(num_classes, 768)
            elif key == 'head.fc.bias':
                value = torch.zeros(num_classes)
            else:
                value = torch.arange(2., dtype=torch.float32) if gamma else torch.zeros(2)
            name, reshape = strict.target_key(key, strict.DEPTHS)
            source[key] = value
            converted[name] = value.reshape(-1, 1, 1).contiguous() if reshape else value
            expected[name] = converted[name].clone()
        self.assertEqual(len(source), 344)
        rows = strict.small22k_mapping(source, converted, expected, num_classes)
        self.assertEqual(len(rows), 344)
        self.assertEqual(sum(row['gamma_reshape'] for row in rows), 36)
        self.assertEqual(tuple(expected['classifier.2.weight'].shape), (num_classes, 768))
        self.assertEqual(tuple(expected['classifier.2.bias'].shape), (num_classes,))
        self.assertIn('classifier.0.weight', expected)
        self.assertIn('classifier.0.bias', expected)
        self.assertTrue(any(row['source'] == 'head.fc.weight'
                            and row['target'] == 'classifier.2.weight'
                            and row['role'] == 'public_K_head_to_be_discarded' for row in rows))
        self.assertEqual(strict.config_num_classes(
            {'num_classes': num_classes,
             'pretrained_cfg': {'num_classes': num_classes, 'tag': strict.TAG}}), num_classes)
        with self.assertRaises(ValueError):
            strict.config_num_classes({'num_classes': num_classes,
                                       'pretrained_cfg': {'num_classes': num_classes + 1}})
        self.assertEqual(strict.FORMAT, 'torchvision-small22k-only-full-head-state-dict-v1')
        self.assertEqual(small_source.REPO, 'timm/convnext_small.fb_in22k_ft_in1k')
        self.assertEqual(small_source.REVISION, 'fe6b358aa409414cdb522ae3d43a2c9922f7cc1b')
        self.assertEqual(small_strict.DEPTHS, (3, 3, 27, 3))


if __name__ == '__main__':
    unittest.main()
