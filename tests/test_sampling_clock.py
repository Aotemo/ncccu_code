"""Pinned fresh clock and P4 reference-stream sharding, without images or models."""
from itertools import islice
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain.fresh import resolve_steps_per_cycle


class SamplingClockTest(unittest.TestCase):
    def test_registered_fresh_uses_pinned_clock_while_legacy_keeps_negative_sweep(self):
        planned = {'steps_per_cycle': 800, 'total_updates': 24000,
                   'warmup_updates': 1600, 'stop_after_updates': 24000}
        fresh_spec = {'schedule': dict(planned)}
        for positive, sweep in ((8, 800), (4, 747)):
            with self.subTest(positive_per_batch=positive):
                job = {'experiment91': {'registered': True}, 'effective_batch': 64,
                       'positive_per_batch': positive, 'cycles': 30, 'warmup_cycles': 2}
                # ex.schedule is the external pinned SCIENCE input; the resolver
                # itself and the legacy arithmetic are exercised unmocked.
                with patch('retrain.experiment91.schedule', return_value=planned) as schedule:
                    steps = resolve_steps_per_cycle(job, 44800, fresh_spec=fresh_spec)
                    self.assertEqual(steps, 800)
                    self.assertEqual(job['cycles'] * steps, planned['total_updates'])
                    self.assertEqual(job['warmup_cycles'] * steps, planned['warmup_updates'])
                    schedule.assert_called_once_with()
                with patch('retrain.experiment91.schedule') as schedule:
                    self.assertEqual(resolve_steps_per_cycle(job, 44800), sweep)
                    legacy = {key: value for key, value in job.items() if key != 'experiment91'}
                    self.assertEqual(resolve_steps_per_cycle(legacy, 44800), sweep)
                    self.assertEqual(resolve_steps_per_cycle(legacy, 44800, fresh_spec=fresh_spec), sweep)
                    schedule.assert_not_called()
                self.assertEqual(set(job), {'experiment91', 'effective_batch', 'positive_per_batch',
                                           'cycles', 'warmup_cycles'})

    def test_P4_eight_rank_shards_preserve_reference_stream_across_attempt_800(self):
        from retrain.data import BalancedBatchSampler
        from retrain.distributed import RankBatchSampler, layout
        from retrain.execution_policy import resolve_topology

        config = {'execution': {'devices': ['cpu'] * 8, 'per_rank_microbatch': 0,
                                'evaluation_physical_batch': 32},
                  'jobs': [{'physical_batch': 32, 'effective_batch': 64,
                            'positive_per_batch': 4, 'seed': 20261029}]}
        job, plan, policy = resolve_topology(config, available_accelerators=0)
        self.assertEqual(plan, layout(32, 64, 4, 8))
        self.assertEqual((plan['world_size'], plan['reference_physical_batch'], job['physical_batch'],
                          plan['accumulation'], plan['global_effective_batch'], policy['global_positive_quota']),
                         (8, 32, 8, 1, 64, 4))
        rows = [{'label': 0}] * 137 + [{'label': 1}] * 13
        fit = list(range(len(rows)))
        arguments = (rows, fit, plan['reference_physical_batch'], job['effective_batch'],
                     job['positive_per_batch'], job['seed'])
        # This is one uninterrupted stream starting at attempt zero. Only its
        # four reference microbatches around 799/800 are retained, not images.
        continuous = BalancedBatchSampler(*arguments, 0, 801)
        tail = list(islice(iter(continuous), 2 * 799, 2 * 801))
        expected = [tail[0] + tail[1], tail[2] + tail[3]]
        streams = [iter(RankBatchSampler(*arguments, 799, 2, rank=rank, world_size=8))
                   for rank in range(8)]
        for batch in expected:
            shards = [next(stream) for stream in streams]
            self.assertEqual([len(shard) for shard in shards], [8] * 8)
            self.assertEqual(sum(shards, []), batch)  # Both indices and augmentation seeds.
            quotas = [sum(rows[index]['label'] for index, _ in shard) for shard in shards]
            self.assertEqual(quotas, [2, 0, 0, 0, 2, 0, 0, 0])
            positive = sum(quotas)
            negative = len(batch) - positive
            self.assertEqual((len(batch), positive, negative, positive * negative), (64, 4, 60, 240))
        # A sampler resumed at the evaluation boundary must retain attempt 800,
        # not restart permutations or the augmentation seed stream at zero.
        resumed = [next(iter(RankBatchSampler(*arguments, 800, 1, rank=rank, world_size=8)))
                   for rank in range(8)]
        self.assertEqual(sum(resumed, []), expected[1])
        zero = iter(BalancedBatchSampler(*arguments, 0, 1))
        self.assertNotEqual(expected[1], next(zero) + next(zero))


if __name__ == '__main__':
    unittest.main()
