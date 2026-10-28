"""Future GPU allocation must fit the same cap and consume an actual peer exit."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_site_inputs', ROOT / 'tools/activation/prepare_gpu_site.py')
inputs = importlib.util.module_from_spec(spec); spec.loader.exec_module(inputs)


class GPUSiteInputsTest(unittest.TestCase):
    def test_future_owner_rank_and_control_masks_share_the_twenty_percent_cap(self):
        # Synthetic input capacities only, never runtime defaults.
        resources = {'affinity': list(range(128)), 'cpuset': list(range(128)), 'cpu_max': ['12000000', '100000']}
        value = inputs.partition_cpus(resources, 'owner')
        self.assertEqual(value['effective_cpus'], 120)
        self.assertEqual(len(value['owner_mask']), 24)
        self.assertEqual(len(value['rank_masks']), 8)
        flat = value['control_mask'] + [cpu for mask in value['rank_masks'] for cpu in mask]
        self.assertEqual(sorted(set(flat)), value['owner_mask'])
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(value['loader_workers'], 1)

    def test_insufficient_or_fractional_quota_cannot_borrow_full_affinity(self):
        with self.assertRaises(ValueError):
            inputs.partition_cpus({'affinity': list(range(128)), 'cpuset': list(range(128)),
                                   'cpu_max': ['4400000', '100000']}, 'owner')
        value = inputs.partition_cpus({'affinity': list(range(128)), 'cpuset': list(range(128)),
                                      'cpu_max': ['5990000', '100000']}, 'owner')
        self.assertEqual(len(value['owner_mask']), 11)
        self.assertEqual(value['loader_workers'], 0)

    def test_missing_peer_exit_wrong_uid_or_current_owner_is_rejected(self):
        value = {'exit_code': 0, 'child_exit_code': 0, 'task_tree_exited': True,
                 'affinity_violations': [], 'executor_id': 'peer', 'identity': {'NCCCU_POD_UID': 'uid'}}
        inputs.verify_peer_exit(value, 'uid', 'owner')
        for change in ({'task_tree_exited': False}, {'exit_code': 1}, {'executor_id': 'owner'},
                       {'identity': {'NCCCU_POD_UID': 'old'}}, {'affinity_violations': [{}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                inputs.verify_peer_exit({**value, **change}, 'uid', 'owner')


if __name__ == '__main__':
    unittest.main()
