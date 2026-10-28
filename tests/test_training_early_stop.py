"""Successful-update budget, paired-dev patience and a valid early completion."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import experiment91 as ex
from retrain.ema import commit_pair
from retrain.single91 import select_checkpoint


class TrainingEarlyStopTest(unittest.TestCase):
    def setUp(self):
        self.science = json.loads((ROOT / 'configs/R95-SCIENCE.json').read_text())
        self.job = {**self.science['job'], 'experiment91': {'fixture': True}}
        configured = patch.object(ex, 'science_configuration', return_value=self.science)
        configured.start(); self.addCleanup(configured.stop)

    def state(self):
        return {'evaluated_cycle': 0, 'early_stop': {
            'reference_auroc': None, 'bad_cycles': 0, 'stopped': False},
            'best_metrics': None, 'last_metrics': None,
            'ema_best_metrics': None, 'ema_last_metrics': None}

    def commit(self, state, online, ema):
        cycle = state['evaluated_cycle'] + 1
        pair = {variant: {'auroc': score, 'error_twice': round((1 - score) * 10000), 'ap': .9}
                for variant, score in [('online', online), ('ema', ema)]}
        return commit_pair(state, pair, cycle, cycle * 800, self.job)[0]

    def completion(self, cycle, early):
        return {'status': 'completed', 'successful_updates': cycle * 800,
                'attempted_updates': cycle * 800, 'skipped_updates': 0,
                'cycle': cycle, 'evaluated_cycle': cycle, 'early_stop': early}

    def test_new_budget_preserves_absolute_warmup_and_ema(self):
        self.assertEqual(ex.schedule(), {'steps_per_cycle': 800, 'total_updates': 24000,
                         'warmup_updates': 1600, 'stop_after_updates': 24000})
        self.assertEqual(self.job['cycles'], 30)
        self.assertEqual(self.job['warmup_cycles'], 2)
        self.assertEqual((self.job['patience'], self.job['min_cycles'], self.job['min_delta']), (4, 1, 0.0))

    def test_four_unimproved_rounds_stop_after_one_baseline(self):
        state = self.state()
        for cycle in range(1, 6):
            state = self.commit(state, .9, .91)
            self.assertEqual(state['early_stop']['bad_cycles'], cycle - 1)
            self.assertEqual(state['early_stop']['stopped'], cycle == 5)
        self.assertEqual(ex.validate_training_completion(self.completion(5, state['early_stop'])), 5)

    def test_ema_improvement_resets_patience_and_persisted_state_continues(self):
        state = self.state()
        for _ in range(4):
            state = self.commit(state, .9, .91)
        state = self.commit(state, .9, .910001)
        self.assertEqual(state['early_stop']['bad_cycles'], 0)
        self.assertFalse(state['early_stop']['stopped'])
        restored = json.loads(json.dumps(state))
        for _ in range(4):
            state = self.commit(state, .9, .910001)
            restored = self.commit(restored, .9, .910001)
        self.assertEqual(restored, state)
        self.assertTrue(state['early_stop']['stopped'])

    def test_full_budget_completion_and_selector_cover_actual_completed_rounds(self):
        state = self.state()
        for cycle in range(1, 31):
            state = self.commit(state, .8 + cycle / 1000, .81 + cycle / 1000)
        self.assertFalse(state['early_stop']['stopped'])
        self.assertEqual(ex.validate_training_completion(self.completion(30, state['early_stop'])), 30)
        entries = [{'cycle': cycle, 'variant': variant, 'successful_updates': cycle * 800,
                    'error_twice': 100, 'ap': .9}
                   for cycle in range(1, 6) for variant in ('online', 'ema')]
        selected = select_checkpoint(entries, completed_cycles=5)
        self.assertEqual((selected['cycle'], selected['variant']), (1, 'ema'))
        entries[-1].update(error_twice=99)
        self.assertEqual(select_checkpoint(entries, completed_cycles=5), entries[-1])


if __name__ == '__main__':
    unittest.main()
