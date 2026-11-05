"""Small arithmetic and rejection checks for the fixed F05 soups."""
import importlib.util
from pathlib import Path
import unittest

import torch

source = Path(__file__).resolve().parents[1] / 'tools/diagnostics/f05_model_soup.py'
spec = importlib.util.spec_from_file_location('f05_model_soup', source)
soup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soup)


class SoupTests(unittest.TestCase):
    def test_uniform_mean_includes_head_age_and_float_buffers(self):
        first = {'backbone': torch.tensor([1., 5.]), 'head': torch.tensor([3.]),
                 'age_weight': torch.tensor([1., -1.]), 'running_mean': torch.tensor([2.]),
                 'num_batches_tracked': torch.tensor(7)}
        second = {'backbone': torch.tensor([3., 1.]), 'head': torch.tensor([7.]),
                  'age_weight': torch.tensor([3., 1.]), 'running_mean': torch.tensor([4.]),
                  'num_batches_tracked': torch.tensor(7)}
        result = soup.average_states([first, second])
        self.assertTrue(torch.equal(result['backbone'], torch.tensor([2., 3.])))
        self.assertTrue(torch.equal(result['head'], torch.tensor([5.])))
        self.assertTrue(torch.equal(result['age_weight'], torch.tensor([2., 0.])))
        self.assertTrue(torch.equal(result['running_mean'], torch.tensor([3.])))
        self.assertEqual(result['num_batches_tracked'].item(), 7)
        self.assertEqual(first['backbone'][0].item(), 1.)

    def test_requires_at_least_two_compatible_finite_states(self):
        with self.assertRaises(ValueError):
            soup.average_states([{'x': torch.tensor([1.])}])
        for other in ({'x': torch.tensor([1., 2.])}, {'y': torch.tensor([1.])},
                      {'x': torch.tensor([1.], dtype=torch.float64)}, {'x': torch.tensor([float('nan')])}):
            with self.assertRaises(ValueError):
                soup.average_states([{'x': torch.tensor([1.])}, other])

    def test_nonfloating_state_must_match_exactly(self):
        with self.assertRaises(ValueError):
            soup.average_states([{'count': torch.tensor(1)}, {'count': torch.tensor(2)}])

    def test_streamed_mean_matches_one_shot_and_can_be_read_twice(self):
        states = [{'x': torch.tensor([float(i), -float(i)]), 'count': torch.tensor(5)} for i in range(4)]
        accumulator = soup.StateMean()
        for state in states:
            accumulator.add(state)
        expected = soup.average_states(states)
        self.assertTrue(torch.equal(accumulator.result()['x'], expected['x']))
        self.assertTrue(torch.equal(accumulator.result()['x'], expected['x']))

    def test_parent_identity_must_be_unique(self):
        census = {'members': [{'round_id': 'r120-first'}, {'round_id': 'r121-only'}]}
        self.assertEqual(soup.find_member(census, 'r120')['round_id'], 'r120-first')
        with self.assertRaises(ValueError):
            soup.find_member(census, 'r122')
        census['members'].append({'round_id': 'r120-second'})
        with self.assertRaises(ValueError):
            soup.find_member(census, 'r120')


if __name__ == '__main__':
    unittest.main()
