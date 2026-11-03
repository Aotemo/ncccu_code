"""Registered horizon, schedule and real shared completion-counter behavior."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


class EMAHorizonTest(unittest.TestCase):
    def test_registered_pair_changes_only_SSL_horizon_and_preserves_R107_supervision(self):
        from retrain import experiment91 as ex
        self.assertTrue(hasattr(ex, 'FIT_EMA_HORIZON_SMALL_BINDING'), 'EMA horizon candidates missing')
        from retrain.ema_assignment import EMA_RECIPE, EMA_HORIZON_RECIPE
        from retrain.ssl_stage import validate_recipe
        self.assertEqual(EMA_HORIZON_RECIPE, dict(EMA_RECIPE, version='fit-ema-assignment-9600-v1', updates=9600))
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        for label, seed, order, binding in (
                ('R232', 20261029, 147, ex.FIT_EMA_HORIZON_SMALL_BINDING),
                ('R233', 20261027, 148, ex.FIT_EMA_HORIZON_SMALL_SEED27_BINDING)):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            job = copy.deepcopy(original['job']); job['seed'] = seed
            self.assertEqual(value['job'], job)
            self.assertEqual(value['candidate_profile'], binding)
            self.assertEqual(value['candidate']['experiment_order'], order)
            self.assertEqual(value['ssl_recipe'], EMA_HORIZON_RECIPE)
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            for key in ('public_source', 'paths', 'data_split', 'schedule', 'early_stopping', 'structure'):
                self.assertEqual(value[key], original[key])
            manifest = {'fit_indices': list(range(128)), 'dev_indices': [128], 'audit_indices': []}
            validate_recipe(EMA_HORIZON_RECIPE, job, manifest, 8)
            wrong = dict(EMA_HORIZON_RECIPE, updates=3200)
            with self.assertRaises(ValueError): validate_recipe(wrong, job, manifest, 8)
            wrong = dict(EMA_HORIZON_RECIPE, version='r115-fit-vicreg-9600-v1')
            with self.assertRaises(ValueError): validate_recipe(wrong, job, manifest, 8)

    def test_both_schedules_have_independent_oracles_and_original1600_behavior(self):
        from retrain.ema_assignment import EMA_HORIZON_RECIPE, teacher_momentum, update_teacher
        from retrain.ssl_stage import _factor
        import torch
        from torch import nn
        self.assertEqual(EMA_HORIZON_RECIPE['warmup_updates'], 160)
        for total in (1600, 9600):
            for step in (1, 160, 161, 800, total - 1, total):
                expected_lr = step / 160 if step <= 160 else .01 + .99 * (1 + math.cos(math.pi * (step - 160) / (total - 160))) / 2
                expected_teacher = .9995 + .0005 * (1 - math.cos(math.pi * (step - 1) / (total - 1))) / 2
                self.assertEqual(_factor(step, total_updates=total), expected_lr)
                self.assertEqual(teacher_momentum(step, total_updates=total), expected_teacher)
            self.assertEqual(teacher_momentum(1, total_updates=total), .9995)
            self.assertEqual(teacher_momentum(total, total_updates=total), 1.)
            with self.assertRaises(ValueError): _factor(total + 1, total_updates=total)
            with self.assertRaises(ValueError): teacher_momentum(total + 1, total_updates=total)
        for step in (1, 160, 161, 800, 1599, 1600):
            self.assertEqual(_factor(step), _factor(step, total_updates=1600))
            self.assertEqual(teacher_momentum(step), teacher_momentum(step, total_updates=1600))
        for fn in (_factor, teacher_momentum):
            with self.assertRaises(ValueError): fn(1, total_updates=3200)
        student = nn.Linear(2, 2, bias=False)
        with torch.no_grad(): student.weight.fill_(2.)
        teacher = copy.deepcopy(student).requires_grad_(False)
        with torch.no_grad(): teacher.weight.zero_()
        center = torch.zeros(1, 2)
        returned = update_teacher(teacher, student, center, torch.tensor([[1., 3.], [3., 1.]]), update=800, total_updates=9600)
        self.assertEqual(returned, teacher_momentum(800, total_updates=9600))
        torch.testing.assert_close(teacher.weight, torch.full_like(teacher.weight, 2 * (1 - returned)))
        torch.testing.assert_close(center, torch.full_like(center, .2))
        self.assertIsNone(teacher.weight.grad)

    def test_actual_shared_completion_gate_rejects_cross_budget_partial_or_dev_records(self):
        from retrain.ema_assignment import EMA_RECIPE, EMA_HORIZON_RECIPE
        from retrain.ssl_stage import ssl_phase_completed
        for recipe in (EMA_RECIPE, EMA_HORIZON_RECIPE):
            record = dict(status='completed', recipe=recipe, successful_updates=recipe['updates'],
                          attempted_updates=recipe['updates'], skipped_updates=0, dev_evaluations=0)
            self.assertTrue(ssl_phase_completed(record, recipe))
            for key, change in (('successful_updates', recipe['updates'] - 1),
                                ('attempted_updates', recipe['updates'] + 1), ('skipped_updates', 1),
                                ('dev_evaluations', 1), ('status', 'running'),
                                ('recipe', EMA_HORIZON_RECIPE if recipe is EMA_RECIPE else EMA_RECIPE)):
                with self.subTest(total=recipe['updates'], key=key):
                    self.assertFalse(ssl_phase_completed(dict(record, **{key: change}), recipe))
            other = EMA_HORIZON_RECIPE if recipe is EMA_RECIPE else EMA_RECIPE
            self.assertFalse(ssl_phase_completed(record, other))
        self.assertFalse(ssl_phase_completed(None, EMA_HORIZON_RECIPE))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(max(1, int(os.environ['NCCCU_CPU_THREADS']) // 2))
    torch.set_num_interop_threads(1)
    unittest.main()
