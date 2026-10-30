"""One small DINO/ResNet-50 plain-state fixture; parent A executes it in the guarded Pod."""
import sys
from pathlib import Path
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools/pretraining'
sys.path.insert(0, str(TOOLS))

from dino_resnet_strict import partition_state, strip_prefixes


class DinoResnetSourceTest(unittest.TestCase):
    def test_plain_backbone_and_actual_prefixes(self):
        import torch
        plain = {'conv1.weight': torch.zeros(1)}
        state, prefixes, mapping = strip_prefixes(plain)
        self.assertEqual(set(state), {'conv1.weight'})
        self.assertEqual(prefixes, [])
        self.assertEqual(mapping, {})
        nested = {'module.backbone.conv1.weight': torch.zeros(1)}
        state, prefixes, mapping = strip_prefixes(nested)
        self.assertEqual(set(state), {'conv1.weight'})
        self.assertEqual(prefixes, ['module.', 'backbone.'])
        self.assertEqual(mapping, {'module.backbone.conv1.weight': 'conv1.weight'})
        with self.assertRaises(ValueError):
            strip_prefixes({'module.conv1.weight': torch.zeros(1), 'conv1.weight': torch.zeros(1)})

    def test_missing_counter_is_derived_and_other_missing_blocks(self):
        import torch
        target = {'conv1.weight', 'bn1.num_batches_tracked'}
        unexpected, derived, blocking = partition_state({'conv1.weight': torch.zeros(1)}, target)
        self.assertEqual(unexpected, [])
        self.assertEqual(derived, ['bn1.num_batches_tracked'])
        self.assertEqual(blocking, [])
        unexpected, derived, blocking = partition_state({'bn1.num_batches_tracked': torch.tensor(0)}, target)
        self.assertEqual(blocking, ['conv1.weight'])

    def test_container_and_extra_head_keys_are_format_mismatch(self):
        import torch
        target = {'conv1.weight', 'bn1.num_batches_tracked'}
        state = {'teacher': {'conv1.weight': torch.zeros(1)},
                 'fc.weight': torch.zeros(1), 'head.last_layer.weight_v': torch.zeros(1)}
        unexpected, derived, blocking = partition_state(state, target)
        self.assertEqual(unexpected, ['fc.weight', 'head.last_layer.weight_v', 'teacher'])
        self.assertEqual(derived, ['bn1.num_batches_tracked'])
        self.assertEqual(blocking, ['conv1.weight'])


if __name__ == '__main__':
    unittest.main()
