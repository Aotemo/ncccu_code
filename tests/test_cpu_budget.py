"""A stale allocation or overlapping executor CPUs must never reach heavy work."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cpu_budget_under_test', ROOT / 'tools/environment/cpu_budget.py')
budget = importlib.util.module_from_spec(spec); spec.loader.exec_module(budget)


class CPUBudgetTest(unittest.TestCase):
    def setUp(self):
        # These are synthetic capacities, not an environment default or a live grant.
        self.resources = {'available': list(range(128)), 'max_total_threads': 24}
        self.identity = {'NCCCU_POD_UID': 'synthetic-current-uid'}
        self.value = budget.build_allocation(self.resources, ['executor-a', 'executor-b'],
                                             self.identity, now=1000, valid_seconds=600)

    def test_twenty_percent_is_shared_by_both_executors(self):
        self.assertEqual(self.value['allocations']['executor-a'], {'threads': 12, 'mask': list(range(12))})
        self.assertEqual(self.value['allocations']['executor-b'], {'threads': 12, 'mask': list(range(12, 24))})
        self.assertEqual(budget.validate_allocation(self.value, self.resources, self.identity,
                                                   'executor-a', now=1001)['threads'], 12)
        smaller = {'available': [2, 4, 6, 8, 10], 'max_total_threads': 2}
        value = budget.build_allocation(smaller, ['a', 'b'], self.identity, now=1000, valid_seconds=600)
        self.assertEqual(value['allocations'], {'a': {'threads': 1, 'mask': [2]}, 'b': {'threads': 1, 'mask': [4]}})

    def test_overlapping_masks_missing_owner_and_quota_reduction_fail(self):
        bad = copy.deepcopy(self.value); bad['allocations']['executor-b']['mask'][0] = 0
        with self.assertRaises(ValueError):
            budget.validate_allocation(bad, self.resources, self.identity, 'executor-a', now=1001)
        with self.assertRaises(ValueError):
            budget.validate_allocation(self.value, self.resources, self.identity, 'unknown-executor', now=1001)
        with self.assertRaises(ValueError):
            budget.validate_allocation(self.value, {**self.resources, 'max_total_threads': 20},
                                       self.identity, 'executor-a', now=1001)

    def test_old_uid_future_or_expired_allocation_fail(self):
        for identity, now in [({'NCCCU_POD_UID': 'old-uid'}, 1001), (self.identity, 999), (self.identity, 1601)]:
            with self.subTest(now=now), self.assertRaises(ValueError):
                budget.validate_allocation(self.value, self.resources, identity, 'executor-a', now=now)

    def test_inadequate_capacity_or_missing_executor_is_rejected(self):
        for owners, resources in [(['only-one'], self.resources), (['a', 'a'], self.resources),
                                   (['a', 'b'], {'available': [1], 'max_total_threads': 1})]:
            with self.subTest(owners=owners), self.assertRaises(ValueError):
                budget.build_allocation(resources, owners, self.identity, now=1000, valid_seconds=600)

    def test_cpuset_parser_keeps_holes_and_rejects_invalid_ranges(self):
        self.assertEqual(budget.parse_cpuset('0-2,7,9-10'), [0, 1, 2, 7, 9, 10])
        for invalid in ('', '5-2', '-1', '1-2-3'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                budget.parse_cpuset(invalid)


if __name__ == '__main__':
    unittest.main()
