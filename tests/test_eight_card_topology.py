"""The eighth rank must shard the same reference stream, without changing G64/P8."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'runtime/bundle')]

from n08_topology import resolve_n08_topology
from retrain.data import BalancedBatchSampler
from retrain.distributed import RankBatchSampler


class EightCardTopologyTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            'execution': {'devices': ['cuda:' + str(n) for n in range(8)],
                          'per_rank_microbatch': 0, 'evaluation_physical_batch': 32},
            'jobs': [{'physical_batch': 32, 'effective_batch': 64, 'positive_per_batch': 8}],
        }

    def test_eight_ranks_keep_global_batch_and_reference_size(self):
        job, topology, policy = resolve_n08_topology(
            self.config, available_accelerators=8, expected_world=8)
        self.assertEqual((topology['world_size'], topology['reference_physical_batch'],
                          job['physical_batch'], topology['accumulation'],
                          topology['global_effective_batch'], topology['global_positive_quota']),
                         (8, 32, 8, 1, 64, 8))
        self.assertEqual(policy['evaluation_physical_batch'], 32)

    def test_missing_device_wrong_world_or_changed_science_is_rejected(self):
        cases = []
        wrong = copy.deepcopy(self.config); wrong['execution']['devices'] = wrong['execution']['devices'][:4]
        cases.append(wrong)
        wrong = copy.deepcopy(self.config); wrong['execution']['devices'][7] = 'cuda:0'
        cases.append(wrong)
        wrong = copy.deepcopy(self.config); wrong['jobs'][0]['physical_batch'] = 64
        cases.append(wrong)
        wrong = copy.deepcopy(self.config); wrong['jobs'][0]['positive_per_batch'] = 4
        cases.append(wrong)
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_n08_topology(config, available_accelerators=8, expected_world=8)
        with self.assertRaises(ValueError):
            resolve_n08_topology(self.config, available_accelerators=7, expected_world=8)

    def test_four_and_eight_rank_reassembly_preserves_indices_and_augmentation_seeds(self):
        rows = [{'label': 0}] * 137 + [{'label': 1}] * 13
        fit = list(range(len(rows)))
        for start in (0, 1599, 1600, 1601, 6400):
            reference = iter(BalancedBatchSampler(rows, fit, 32, 64, 8, 20261027, start, 2))
            streams = {
                world: [iter(RankBatchSampler(rows, fit, 32, 64, 8, 20261027, start, 2,
                                               rank=rank, world_size=world))
                        for rank in range(world)]
                for world in (4, 8)
            }
            for _ in range(2):
                expected = next(reference) + next(reference)
                for world in (4, 8):
                    shards = [next(stream) for stream in streams[world]]
                    self.assertEqual(sum(shards, []), expected)
                    self.assertEqual(sum(rows[index]['label'] for index, _ in sum(shards, [])), 8)
                    quotas = [sum(rows[index]['label'] for index, _ in shard) for shard in shards]
                    self.assertEqual(quotas, [4, 0, 4, 0] if world == 4 else [4, 0, 0, 0, 4, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
